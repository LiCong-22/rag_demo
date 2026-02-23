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
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_BASE_URL
)

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
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.retriever_with_sources = self.vector_store.as_retriever(search_kwargs={"k": 3})

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

    def hybrid_search(self, query: str, k: int = 5):
        """混合检索：向量 + BM25"""
        # 向量检索
        vector_results = self.vector_store.similarity_search(query, k=k)

        # BM25 检索
        bm25_results = self._bm25_search(query, k=k)

        # RRF 融合
        fused_results = self._rrf_fusion([vector_results, bm25_results], k=60)

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
        docs = self.hybrid_search(question, k=5)
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

engine = RAGEngine()