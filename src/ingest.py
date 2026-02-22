# src/ingest.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import MILVUS_URI, COLLECTION_NAME, DATA_PATH, EMBEDDING_MODEL_PATH

def run_ingestion():
    print(">>> 开始加载文档...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据目录不存在：{DATA_PATH}")
        return

    docs = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(('.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs.extend(loader.load())
                    print(f"  ✓ 加载：{file}")
                except Exception as e:
                    print(f"  ⚠️ 跳过 {file}: {e}")
    
    if len(docs) == 0:
        print("❌ 未加载到任何文档")
        return

    print(f">>> 共加载 {len(docs)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f">>> 分块后得到 {len(splits)} 个向量块")

    # 初始化 Embedding (本地加载)
    print(">>> 加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 测试 Milvus 连接
    print(">>> 测试 Milvus 连接...")
    try:
        from pymilvus import connections
        connections.connect(uri=MILVUS_URI)
        print("✅ Milvus 连接成功")
    except Exception as e:
        print(f"❌ Milvus 连接失败：{e}")
        return

    print(">>> 正在向量化并存入 Milvus...")
    try:
        vector_store = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )
        print(">>> ✅ 入库完成！")
    except Exception as e:
        print(f"❌ 入库失败：{e}")

if __name__ == "__main__":
    run_ingestion()