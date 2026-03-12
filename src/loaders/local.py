# src/loaders/local.py
import os
from typing import List
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

    def _get_loader(self, file_path: str):
        """根据文件扩展名选择合适的加载器"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            from langchain_community.document_loaders import PyMuPDFLoader
            return PyMuPDFLoader(file_path)
        elif ext == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(file_path)
        elif ext in ['.xlsx', '.xls']:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            return UnstructuredExcelLoader(file_path)
        else:
            # 默认为 TextLoader (支持 .md, .txt 等)
            from langchain_community.document_loaders import TextLoader
            return TextLoader(file_path, encoding='utf-8')

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
                        loader = self._get_loader(file_path)
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
