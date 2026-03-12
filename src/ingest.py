# src/ingest.py
import os
import uuid
import re
import time
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    MILVUS_URI, COLLECTION_NAME, DATA_PATH, EMBEDDING_MODEL_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, ENABLE_PARENT_CHILD,
    DATA_SOURCE
)
from src.loaders import load_all_docs

# 计时装饰器
def timer(name: str):
    """计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            print(f"\n[⏱️] {name} 开始...")
            result = func(*args, **kwargs)
            print(f"[⏱️] {name} 完成，耗时: {time.time() - start:.2f}秒")
            return result
        return wrapper
    return decorator

def split_with_parent_child(documents: List[Document]) -> List[Document]:
    """
    父子分块策略：
    - 子块：基于段落拆分，保留最小语义单元（200-400字符）
    - 父块：多个子块的组合，形成完整章节（800+字符）
    """
    print(">>> 开始父子分块...")

    # 先按章节拆分（识别标题）
    chapter_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE * 2,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", ".", "!", "?"]
    )

    all_chunks = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        # 先拆分成较大的章节块
        chapters = chapter_splitter.split_text(doc.page_content)

        # 对每个章节再做细粒度分块（子块）
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )

        parent_chunks = []
        parent_id = None  # 将在创建父块时设置

        for chapter in chapters:
            # 对章节进行细粒度分块
            children = child_splitter.split_text(chapter)

            for child in children:
                if len(child.strip()) < 50:  # 跳过太短的块
                    continue

                child_id = str(uuid.uuid4())

                # 创建子块 (暂时先不设置 parent_id，等父块创建后再更新)
                child_doc = Document(
                    page_content=child,
                    metadata={
                        "source": source,
                        "chunk_id": child_id,
                        "parent_id": "",  # 暂时为空，后续填充
                        "level": "child",
                        "title": extract_title(chapter)
                    }
                )
                all_chunks.append(child_doc)
                parent_chunks.append((child_id, child))  # 保存 (child_id, content) 对

            # 合并多个子块形成父块
            if parent_chunks:
                # 将相邻子块合并为父块（每3-5个一组）
                for i in range(0, len(parent_chunks), 3):
                    group = parent_chunks[i:i+5]
                    if group:
                        parent_content = "\n".join([c[1] for c in group])
                        parent_id = str(uuid.uuid4())  # 父块 ID

                        # 创建父块
                        parent_doc = Document(
                            page_content=parent_content,
                            metadata={
                                "source": source,
                                "chunk_id": parent_id,
                                "parent_id": "",  # 父块没有父级
                                "level": "parent",
                                "title": extract_title(group[0][1] if group else "")
                            }
                        )
                        all_chunks.append(parent_doc)

                        # 更新所有子块的 parent_id 指向这个父块
                        for child_id, _ in group:
                            # 找到对应的子块并更新 parent_id
                            for doc in all_chunks:
                                if doc.metadata.get("chunk_id") == child_id:
                                    doc.metadata["parent_id"] = parent_id
                                    break

                parent_chunks = []

    print(f">>> 父子分块完成：共 {len(all_chunks)} 个块")
    return all_chunks


def extract_title(text: str) -> str:
    """从文本中提取标题"""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            return line.lstrip('#').strip()
        if line and len(line) < 100 and not line.endswith(('.', '，', '、', ';', '：')):
            return line
    return "无标题"


def run_ingestion():
    total_start = time.time()

    # 1. 加载文档
    print("\n" + "="*50)
    print("[1/4] 加载文档...")
    docs = load_all_docs(DATA_SOURCE)
    print(f"    共加载 {len(docs)} 个文档")

    if len(docs) == 0:
        print("❌ 未加载到任何文档")
        return

    # 2. 分块处理
    print("\n" + "="*50)
    print("[2/4] 分块处理...")
    chunk_start = time.time()
    if ENABLE_PARENT_CHILD:
        splits = split_with_parent_child(docs)
    else:
        # 传统固定大小分块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
    print(f"    获得 {len(splits)} 个块，耗时: {time.time() - chunk_start:.2f}秒")

    # 3. 加载模型 & 连接 Milvus
    print("\n" + "="*50)
    print("[3/4] 加载模型 & 连接 Milvus...")
    model_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"    模型加载耗时: {time.time() - model_start:.2f}秒")

    # 测试 Milvus 连接
    milvus_start = time.time()
    try:
        from pymilvus import connections, utility
        # 使用 host/port 方式连接，避免异步问题
        milvus_host = "localhost"
        milvus_port = "19530"
        if "://" in MILVUS_URI:
            # 处理 milvus://host:port 格式
            uri_parts = MILVUS_URI.split("://")[1].split(":")
            milvus_host = uri_parts[0]
            milvus_port = uri_parts[1] if len(uri_parts) > 1 else "19530"
        connections.connect(host=milvus_host, port=milvus_port)
        print(f"    Milvus 连接耗时: {time.time() - milvus_start:.2f}秒")

        # 删除旧集合（如果存在）
        if utility.has_collection(COLLECTION_NAME):
            print(f"    删除旧集合: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Milvus 连接失败：{e}")
        return

    # 4. 向量化入库
    print("\n" + "="*50)
    print("[4/4] 向量化并存入 Milvus...")
    vector_start = time.time()
    try:
        vector_store = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            connection_args={"host": milvus_host, "port": milvus_port},
            collection_name=COLLECTION_NAME,
        )
        print(f"    向量化入库耗时: {time.time() - vector_start:.2f}秒")
    except Exception as e:
        print(f"❌ 入库失败：{e}")
        return

    # 总耗时
    print("\n" + "="*50)
    print(f"✅ 全部完成！总耗时: {time.time() - total_start:.2f}秒")
    print("="*50)

if __name__ == "__main__":
    run_ingestion()
