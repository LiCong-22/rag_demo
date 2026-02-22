# src/engine.py
import torch
import time
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    MILVUS_URI, COLLECTION_NAME,
    LLM_MODEL_PATH, EMBEDDING_MODEL_PATH,
    LLM_TYPE, OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_BASE_URL
)

class RAGEngine:
    def __init__(self):
        print(">>> [1/3] 初始化 Embedding 模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(">>> [2/3] 连接 Milvus...")
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )

        print(f">>> [3/3] 加载 LLM 模型 ({LLM_TYPE})...")

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

        # 计时：向量检索
        start_retrieval = time.time()
        docs = self.retriever_with_sources.invoke(question)
        retrieval_time = time.time() - start_retrieval

        # 计时：LLM 生成
        start_llm = time.time()
        answer = self.rag_chain.invoke(question)
        llm_time = time.time() - start_llm

        total_time = time.time() - start_total

        print(f">>> 检索耗时: {retrieval_time:.2f}s (找到 {len(docs)} 个文档)")
        print(f">>> LLM 生成耗时: {llm_time:.2f}s")
        print(f">>> 总耗时: {total_time:.2f}s")

        return {
            "answer": answer,
            "sources": [doc.page_content for doc in docs]
        }

engine = RAGEngine()