# src/loaders/__init__.py
from typing import List
from langchain_core.documents import Document

from .base import BaseLoader
from .local import LocalFileLoader
from .confluence import ConfluenceLoader_


def get_loader(data_source: str = "local", **kwargs) -> BaseLoader:
    """
    根据配置获取对应的加载器

    Args:
        data_source: 数据源类型 ("local", "confluence", "all")
        **kwargs: 传递给加载器的参数

    Returns:
        加载器实例或加载器列表
    """
    from src import config

    if data_source == "local":
        return LocalFileLoader(
            data_path=kwargs.get("data_path", config.DATA_PATH)
        )
    elif data_source == "confluence":
        return ConfluenceLoader_(
            url=kwargs.get("url", config.CONFLUENCE_URL),
            username=kwargs.get("username", config.CONFLUENCE_USERNAME),
            api_key=kwargs.get("api_key", config.CONFLUENCE_API_KEY),
            space_key=kwargs.get("space_key", config.CONFLUENCE_SPACE_KEY)
        )
    elif data_source == "all":
        # 返回加载器列表
        return [
            LocalFileLoader(data_path=config.DATA_PATH),
            ConfluenceLoader_(
                url=config.CONFLUENCE_URL,
                username=config.CONFLUENCE_USERNAME,
                api_key=config.CONFLUENCE_API_KEY,
                space_key=config.CONFLUENCE_SPACE_KEY
            )
        ]
    else:
        raise ValueError(f"不支持的数据源类型: {data_source}")


def load_all_docs(data_source: str = "all") -> List[Document]:
    """
    加载所有数据源的文档

    Args:
        data_source: 数据源类型

    Returns:
        所有加载的文档列表
    """
    loader = get_loader(data_source)

    if isinstance(loader, list):
        # 多个加载器
        docs = []
        for l in loader:
            # 跳过空的 Confluence 配置
            if isinstance(l, ConfluenceLoader_) and not l.url:
                print("  ⚠️ Confluence URL 未配置，跳过加载")
                continue
            if isinstance(l, ConfluenceLoader_):
                # 检查依赖
                try:
                    from atlassian import Confluence
                except ImportError:
                    print("  ⚠️ 缺少 atlassian-python-api 包，跳过 Confluence 加载")
                    print("     安装命令: pip install atlassian-python-api")
                    continue
            docs.extend(l.load())
        return docs
    else:
        # 单个加载器
        return loader.load()


__all__ = [
    "BaseLoader",
    "LocalFileLoader",
    "ConfluenceLoader_",
    "get_loader",
    "load_all_docs"
]
