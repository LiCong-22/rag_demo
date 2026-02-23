# src/loaders/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseLoader(ABC):
    """文档加载器基类"""

    @abstractmethod
    def load(self) -> List[Document]:
        """加载文档"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """数据源名称"""
        pass
