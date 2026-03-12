# src/engine.py
import torch
import time
import numpy as np
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
import jieba
from src.config import (
    MILVUS_URI, COLLECTION_NAME,
    LLM_MODEL_PATH, EMBEDDING_MODEL_PATH,
    LLM_TYPE, OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_BASE_URL,
    ENABLE_HYDE, ENABLE_QUERY_EXPANSION,
    EXPANSION_COUNT, RETRIEVAL_K,
    ENABLE_PARENT_CHILD,
    ENABLE_WEB_SEARCH
)

# ==================== 动态配置 ====================
# 这些变量可以在运行时修改
_enable_hyde = ENABLE_HYDE
_enable_query_expansion = ENABLE_QUERY_EXPANSION

def get_rag_config():
    """获取当前 RAG 配置"""
    return {
        "enable_hyde": _enable_hyde,
        "enable_query_expansion": _enable_query_expansion,
        "expansion_count": EXPANSION_COUNT,
        "retrieval_k": RETRIEVAL_K
    }

def set_rag_config(enable_hyde: bool = None, enable_query_expansion: bool = None):
    """动态修改 RAG 配置"""
    global _enable_hyde, _enable_query_expansion
    if enable_hyde is not None:
        _enable_hyde = enable_hyde
    if enable_query_expansion is not None:
        _enable_query_expansion = enable_query_expansion
    return get_rag_config()

class RAGEngine:
    def __init__(self):
        print(">>> [1/4] 初始化 Embedding 模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print(">>> [2/4] 连接 Milvus...")
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )

        print(">>> [3/4] 初始化 BM25 检索器...")
        self._init_bm25()

        # 初始化父子分块映射
        if ENABLE_PARENT_CHILD:
            print(">>> [3.5/4] 初始化父子分块索引...")
            self._init_parent_child_index()

        print(f">>> [4/4] 加载 LLM 模型 ({LLM_TYPE})...")

        # 根据 LLM_TYPE 选择不同的初始化方式
        if LLM_TYPE == "local":
            self.llm = self._init_local_llm()
        elif LLM_TYPE == "openai":
            self.llm = self._init_openai_llm()
        elif LLM_TYPE == "anthropic":
            self.llm = self._init_anthropic_llm()
        else:
            raise ValueError(f"不支持的 LLM_TYPE: {LLM_TYPE}")
        
        # 构建 RAG 链
        template = """你是一个汽车电子软件研发助手。请根据以下已知信息回答用户问题。

        要求：
        1. 只回答一次，不要重复相同内容
        2. 如果已知信息中不包含答案，请直接说"知识库中未找到相关信息"
        3. 回答简洁专业，不要编造

        已知信息：
        {context}

        用户问题：
        {question}

        回答：
        """
        prompt = PromptTemplate.from_template(template)
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.retriever_with_sources = self.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

        print(">>> ✅ RAG 引擎初始化完成！")

    def _init_bm25(self):
        """初始化 BM25 检索器"""
        # 从 Milvus 获取所有文档用于构建 BM25 索引
        all_docs = self.vector_store.similarity_search("", k=10000)
        self.bm25_corpus = [doc.page_content for doc in all_docs]
        self.bm25_docs = all_docs  # 保留原始文档引用

        # 中文分词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"    BM25 索引构建完成，共 {len(self.bm25_corpus)} 个文档")

    def _init_parent_child_index(self):
        """初始化父子分块索引，构建 chunk_id -> 父块 的映射"""
        if not ENABLE_PARENT_CHILD:
            return

        print("    构建父子分块映射...")
        # 获取所有文档
        all_docs = self.vector_store.similarity_search("", k=16384)

        # 构建映射
        # chunk_id -> Document
        self.chunk_map = {}
        # parent_id -> Document (父块)
        self.parent_map = {}

        for doc in all_docs:
            chunk_id = doc.metadata.get("chunk_id")
            parent_id = doc.metadata.get("parent_id")
            level = doc.metadata.get("level", "parent")

            if chunk_id:
                self.chunk_map[chunk_id] = doc

                # 如果是父块，建立 parent_id -> doc 映射
                if level == "parent" or not parent_id:
                    self.parent_map[chunk_id] = doc

        print(f"    父子分块索引构建完成: {len(self.chunk_map)} 个块, {len(self.parent_map)} 个父块")

    def _resolve_parent_child(self, docs: list) -> list:
        """
        解析父子块关系：
        - 如果检索到子块，替换为对应的父块
        - 如果检索到父块，直接返回
        """
        if not ENABLE_PARENT_CHILD:
            return docs

        resolved_docs = []
        seen_content = set()

        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            parent_id = doc.metadata.get("parent_id")
            level = doc.metadata.get("level", "parent")

            # 如果是子块，尝试获取父块
            if level == "child" and parent_id:
                # 尝试通过 parent_id 找到父块
                parent_doc = self.parent_map.get(parent_id)
                if parent_doc:
                    content = parent_doc.page_content
                    if content not in seen_content:
                        resolved_docs.append(parent_doc)
                        seen_content.add(content)
                    continue

            # 如果是父块或找不到父块，直接使用原块
            content = doc.page_content
            if content not in seen_content:
                resolved_docs.append(doc)
                seen_content.add(content)

        return resolved_docs

    def _bm25_search(self, query: str, k: int = 5):
        """BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.bm25_docs[i] for i in top_indices if scores[i] > 0]

    def _rrf_fusion(self, results_list: list, k: int = 60):
        """RRF (Reciprocal Rank Fusion) 混合检索算法"""
        doc_scores = {}

        for results in results_list:
            for rank, doc in enumerate(results):
                doc_key = doc.page_content
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0}
                doc_scores[doc_key]["score"] += 1.0 / (k + rank + 1)

        # 按分数排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def _extract_text(self, result) -> str:
        """从 LLM 返回结果中提取文本"""
        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            return item.get('text', '')
            elif isinstance(content, str):
                return content
        return str(result)

    def _generate_hypothetical_doc(self, question: str) -> str:
        """生成假设文档 (HyDE)"""
        prompt = f"""请根据问题生成一个可能包含答案的假设文档片段。
要求：直接给出假设文档内容，不要有任何前缀解释。问题越简洁越好。

问题：{question}
假设文档："""
        try:
            result = self.llm.invoke(prompt)
            return self._extract_text(result)
        except Exception as e:
            print(f"    ⚠️ HyDE 生成失败: {e}")
            return ""

    def _expand_query(self, question: str, num_expansions: int = None) -> list[str]:
        """生成同义查询 (查询扩展)"""
        if num_expansions is None:
            num_expansions = EXPANSION_COUNT

        prompt = f"""生成 {num_expansions} 个与以下问题意思相同但表述不同的问法。
要求：
1. 每行一个问法，不要有编号或前缀
2. 直接返回问法列表，不要有任何解释

原始问题：{question}
同义问法："""
        try:
            result = self.llm.invoke(prompt)
            expanded = self._extract_text(result).strip().split('\n')
            # 过滤空行并返回
            expanded = [q.strip() for q in expanded if q.strip()]
            return [question] + expanded[:num_expansions]
        except Exception as e:
            print(f"    ⚠️ 查询扩展失败: {e}")
            return [question]

    def _hyde_search(self, question: str, k: int) -> list:
        """使用 HyDE 进行检索"""
        hypothetical_doc = self._generate_hypothetical_doc(question)
        if hypothetical_doc:
            print(f"    📝 HyDE 假设文档: {hypothetical_doc[:100]}...")
            return self.vector_store.similarity_search(hypothetical_doc, k=k)
        return []

    def _expanded_search(self, question: str, k: int) -> list:
        """使用查询扩展进行检索"""
        expanded_queries = self._expand_query(question)
        if len(expanded_queries) > 1:
            print(f"    🔍 扩展查询: {expanded_queries}")

        all_results = []
        for query in expanded_queries:
            results = self.vector_store.similarity_search(query, k=k)
            all_results.append(results)

        # 合并所有结果
        merged = []
        seen = set()
        for results in all_results:
            for doc in results:
                key = doc.page_content
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    def hybrid_search(self, query: str, k: int = None):
        """混合检索：向量 + BM25 + HyDE + 查询扩展"""
        if k is None:
            k = RETRIEVAL_K

        print(f">>> 开始检索 (HyDE={_enable_hyde}, 扩展={_enable_query_expansion})")

        all_results = []

        # 1. 基础向量检索
        vector_results = self.vector_store.similarity_search(query, k=k)
        all_results.append(vector_results)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, k=k)
        all_results.append(bm25_results)

        # 3. HyDE 检索 (如果启用)
        if _enable_hyde:
            hyde_results = self._hyde_search(query, k)
            if hyde_results:
                all_results.append(hyde_results)

        # 4. 查询扩展检索 (如果启用)
        if _enable_query_expansion:
            expanded_results = self._expanded_search(query, k)
            if expanded_results:
                all_results.append(expanded_results)

        # RRF 融合所有结果
        fused_results = self._rrf_fusion(all_results, k=60)

        # 父子块解析：如果检索到子块，替换为父块
        if ENABLE_PARENT_CHILD:
            fused_results = self._resolve_parent_child(fused_results)

        return fused_results[:k]

    def _init_local_llm(self):
        """初始化本地 HuggingFace 模型"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        print(f"    模型设备：{next(model.parameters()).device}")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

        return HuggingFacePipeline(pipeline=pipe)

    def _init_openai_llm(self):
        """初始化 OpenAI API 模型"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.1,
            api_key=OPENAI_API_KEY,
        )

    def _init_anthropic_llm(self):
        """初始化 Anthropic Claude 模型 (支持 MiniMax 等兼容 API)"""
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=0.1,
            anthropic_api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
        )

    def query(self, question: str):
        start_total = time.time()
        print(f">>> 正在处理：{question}")

        # 计时：混合检索
        start_retrieval = time.time()
        docs = self.hybrid_search(question)
        retrieval_time = time.time() - start_retrieval

        # 构建 context
        context = "\n\n".join(doc.page_content for doc in docs)

        # 计时：LLM 生成
        start_llm = time.time()
        answer = self.llm.invoke(f"已知信息：\n{context}\n\n用户问题：{question}\n\n回答：")
        llm_time = time.time() - start_llm

        total_time = time.time() - start_total

        print(f">>> 检索耗时: {retrieval_time:.2f}s (找到 {len(docs)} 个文档)")
        print(f">>> LLM 生成耗时: {llm_time:.2f}s")
        print(f">>> 总耗时: {total_time:.2f}s")

        # 处理 LLM 返回值
        answer_text = ""
        thinking_text = ""

        try:
            if hasattr(answer, 'content'):
                content = answer.content
                # 处理 MiniMax/Claude 的思考模型返回格式 (list with type field)
                if isinstance(content, list) and len(content) > 0:
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get('type', '')
                            if item_type == 'text':
                                answer_text = item.get('text', '')
                            elif item_type == 'thinking':
                                thinking_text = item.get('thinking', '')
                            elif item_type == 'output':
                                # 有时是 output 字段
                                answer_text = item.get('text', item.get('output', str(item)))
                            else:
                                # 普通模型可能返回其他格式，尝试直接提取
                                answer_text = str(item)
                        else:
                            answer_text = str(item)
                elif isinstance(content, str):
                    # 普通字符串格式 (如本地模型)
                    answer_text = content
                elif content is not None:
                    # 其他格式
                    answer_text = str(content)
            elif isinstance(answer, str):
                # 直接是字符串
                answer_text = answer
            else:
                answer_text = str(answer)
        except Exception as e:
            print(f"⚠️ 解析回答时出错: {e}")
            answer_text = str(answer)

        return {
            "answer": answer_text,
            "thinking": thinking_text,
            "sources": [doc.page_content for doc in docs]
        }

# ==================== RAG Agent ====================
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent


class RAGAgent:
    """RAG Agent - 结合知识库检索和实时搜索"""

    def __init__(self):
        # 复用现有 RAG Engine（已初始化 LLM）
        self.rag_engine = engine
        self._setup_tools()

    def _setup_tools(self):
        """设置 Agent Tools"""

        @tool
        def rag_search(query: str) -> str:
            """
            检索知识库中的相关信息。
            当用户询问汽车电子、软件架构、技术细节等问题时使用。
            """
            result = self.rag_engine.hybrid_search(query)
            return self._format_rag_result(result)

        # 创建搜索工具（带异常处理）
        from langchain_core.tools import StructuredTool

        def safe_web_search(query: str) -> str:
            """搜索互联网获取最新信息（带异常处理）"""
            try:
                tool = DuckDuckGoSearchRun()
                return tool.run(query)
            except Exception as e:
                return f"搜索暂时不可用：{str(e)}。请尝试直接回答或使用知识库。"

        # 构建工具列表
        tools_list = [rag_search]

        # 添加网页搜索（如果启用且可用）
        if ENABLE_WEB_SEARCH:
            try:
                self.search_tool = StructuredTool.from_function(
                    func=safe_web_search,
                    name="web_search",
                    description="搜索互联网获取最新信息。当用户询问最新新闻、技术动态等问题时使用。"
                )
                tools_list.append(self.search_tool)
            except Exception:
                print("    ⚠️ 网页搜索工具加载失败")

        # 添加本地工具（不需要网络）
        @tool
        def calculator(expression: str) -> str:
            """
            计算数学表达式。
            当用户询问计算问题时使用，例如：2+3*4, 100/7, sqrt(16)等。
            """
            try:
                import math
                # 安全地计算表达式
                allowed_names = {
                    "abs": abs, "max": max, "min": min, "pow": pow,
                    "round": round, "sum": sum, "len": len,
                    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                    "tan": math.tan, "log": math.log, "log10": math.log10,
                    "pi": math.pi, "e": math.e
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return str(result)
            except Exception as e:
                return f"计算错误：{str(e)}"

        @tool
        def python_repl(code: str) -> str:
            """
            执行 Python 代码。
            当用户要求编写或运行代码时使用。
            """
            try:
                import sys
                from io import StringIO

                # 捕获输出
                old_stdout = sys.stdout
                sys.stdout = StringIO()

                # 执行代码
                exec(code, {"__builtins__": __builtins__})

                # 获取输出
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

                return output if output else "代码执行完成（无输出）"
            except Exception as e:
                return f"执行错误：{str(e)}"

        @tool
        def read_file(file_path: str) -> str:
            """
            读取本地文件内容。
            当用户询问需要查看某个文件的内容时使用。
            注意：只读文件，不要尝试执行或修改。
            """
            import os
            from pathlib import Path

            # 安全检查：禁止读取敏感路径
            forbidden_paths = [
                "/etc/", "/usr/bin/", "/usr/sbin/",
                "C:\\Windows", "C:\\Program Files",
                "C:\\Users\\Admin\\AppData"
            ]

            abs_path = os.path.abspath(file_path)

            # 检查是否包含禁止路径
            for forbidden in forbidden_paths:
                if forbidden in abs_path:
                    return f"错误：禁止访问系统目录 {forbidden}"

            # 检查文件是否存在
            if not os.path.exists(abs_path):
                return f"错误：文件不存在: {file_path}"

            # 检查是否是文件
            if not os.path.isfile(abs_path):
                return f"错误：不是文件: {file_path}"

            # 限制文件大小 (最大 1MB)
            file_size = os.path.getsize(abs_path)
            if file_size > 1024 * 1024:
                return f"错误：文件太大 ({file_size} bytes)，最大支持 1MB"

            # 限制文件类型（只允许文本文件）
            allowed_extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.xml', '.csv', '.log', '.conf', '.cfg', '.ini', '.html', '.css', '.sql', '.sh', '.bat', '.ps1']
            ext = Path(file_path).suffix.lower()
            if ext not in allowed_extensions:
                return f"错误：不支持的文件类型: {ext}，只允许: {', '.join(allowed_extensions)}"

            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"文件: {file_path}\n内容:\n{content}"
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(abs_path, 'r', encoding='gbk') as f:
                        content = f.read()
                    return f"文件: {file_path}\n内容:\n{content}"
                except Exception as e:
                    return f"错误：无法读取文件编码: {str(e)}"
            except Exception as e:
                return f"错误：读取失败: {str(e)}"

        tools_list.append(python_repl)
        tools_list.append(calculator)
        tools_list.append(read_file)

        self.tools = tools_list

    def _format_rag_result(self, result):
        """格式化 RAG 结果"""
        if not result:
            return "知识库中未找到相关信息"
        if isinstance(result, list):
            if not result:
                return "知识库中未找到相关信息"
            context = "\n\n".join(doc.page_content for doc in result)
            sources = [doc.page_content[:200] + "..." for doc in result]
            return f"知识库检索结果：\n{context}\n\n来源：{sources}"
        return str(result)

    def query(self, question: str) -> dict:
        """Agent 对话入口 - 复用现有 LLM 配置"""
        import json

        # 使用 RAG Engine 中已初始化的 LLM
        llm = self.rag_engine.llm

        # 创建 Agent (使用 langchain 0.3.x 新 API)
        agent = create_agent(llm, self.tools)

        print(f"\n{'='*60}")
        print(f">>> 🤖 Agent 开始处理问题: {question}")
        print(f">>> 🛠️ 可用工具: {[t.name for t in self.tools]}")
        print(f"{'='*60}\n")

        # 执行（使用 stream 模式获取中间步骤）
        result = agent.invoke({"messages": [("user", question)]})

        # 解析 Agent 返回的消息
        messages = result.get("messages", [])

        # 打印 Agent 执行过程
        print(f"\n{'='*60}")
        print(">>> 📋 Agent 执行过程:")
        print(f"{'='*60}")

        # 提取思考过程和最终回答
        thinking_parts = []
        final_answer = ""
        tool_calls = []

        for i, msg in enumerate(messages):
            msg_type = getattr(msg, 'type', 'unknown')
            print(f"\n--- 步骤 {i}: {msg_type} ---")

            # 处理不同格式的消息
            if msg_type == 'human':
                print(f"用户问题: {getattr(msg, 'content', '')}")
                continue
            elif msg_type == 'ai':
                # AI 消息，可能是思考或最终回答
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                item_type = item.get('type', '')
                                if item_type == 'thinking':
                                    thinking = item.get('text', '')
                                    print(f"💭 思考: {thinking[:200]}...")
                                    thinking_parts.append(thinking)
                                elif item_type == 'text':
                                    text = item.get('text', '')
                                    print(f"📝 文本: {text[:200]}...")
                                    final_answer = text
                                elif item_type == 'tool_use':
                                    tool_name = item.get('name', '')
                                    tool_input = item.get('input', '')
                                    print(f"🔧 调用工具: {tool_name}")
                                    print(f"   输入: {str(tool_input)[:200]}...")
                                    tool_calls.append({"tool": tool_name, "input": tool_input})
                            else:
                                # 可能是最终回答
                                final_answer = str(item)
                    elif isinstance(content, str):
                        print(f"📝 文本: {content[:200]}...")
                        final_answer = content

                # 打印工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"🔧 调用工具: {tc.get('name', 'unknown')}")
                        print(f"   输入: {str(tc.get('args', {}))[:200]}...")

            elif msg_type == 'tool':
                # 工具返回结果
                if hasattr(msg, 'content'):
                    print(f"📤 工具返回: {str(msg.content)[:200]}...")

        print(f"\n{'='*60}")
        print(">>> ✅ Agent 执行完成")
        print(f"{'='*60}\n")

        # 合并思考过程
        thinking = "\n".join(thinking_parts)

        return {
            "answer": final_answer,
            "thinking": thinking,
            "sources": []
        }


# 保留原有的 engine 实例
engine = RAGEngine()

# 全局 RAG Agent 实例（在 engine 初始化后）
rag_agent = RAGAgent()