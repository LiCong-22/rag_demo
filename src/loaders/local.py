# src/loaders/local.py
import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from .base import BaseLoader


class LocalFileLoader(BaseLoader):
    """本地文件加载器"""

    def __init__(self, data_path: str, file_types: List[str] = None):
        """
        初始化本地文件加载器

        Args:
            data_path: 数据目录路径
            file_types: 支持的文件扩展名列表
        """
        self.data_path = data_path
        self.file_types = file_types or ['.md', '.txt']

    def load(self) -> List[Document]:
        """加载本地文件"""
        docs = []

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.file_types):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        loaded_docs = loader.load()
                        # 添加source元数据
                        for d in loaded_docs:
                            d.metadata["source"] = file
                        docs.extend(loaded_docs)
                        print(f"  ✓ 加载本地文件：{file}")
                    except Exception as e:
                        print(f"  ⚠️ 跳过 {file}: {e}")

        return docs

    def get_source_name(self) -> str:
        return "local"
