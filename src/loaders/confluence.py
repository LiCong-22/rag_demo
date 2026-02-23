# src/loaders/confluence.py
from typing import List, Optional
from urllib.parse import quote
from langchain_core.documents import Document
from .base import BaseLoader


class ConfluenceLoader_(BaseLoader):
    """Confluence 文档加载器"""

    def __init__(
        self,
        url: str,
        username: str,
        api_key: str,
        space_key: Optional[str] = None,
        page_ids: Optional[List[str]] = None
    ):
        """
        初始化 Confluence 加载器

        Args:
            url: Confluence 实例 URL (如 https://your-domain.atlassian.net)
            username: Confluence 用户名/邮箱
            api_key: Confluence API Token
            space_key: Confluence 空间键 (可选)
            page_ids: 指定页面 ID 列表 (可选)
        """
        self.url = url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.page_ids = page_ids
        self._client = None

    def _get_client(self):
        """获取 Confluence 客户端"""
        if self._client is None:
            from atlassian import Confluence
            self._client = Confluence(
                url=self.url,
                username=self.username,
                password=self.api_key
            )
        return self._client

    def _get_space_key(self, client) -> Optional[str]:
        """获取正确的 space key"""
        if not self.space_key:
            return None

        try:
            spaces = client.get_all_spaces()
            if isinstance(spaces, dict):
                results = spaces.get('results', [])
                print(f"     可用空间: {[(s.get('key'), s.get('name')) for s in results]}")

                # 精确匹配 key 或 name
                for s in results:
                    key = s.get('key', '')
                    name = s.get('name', '')
                    if key == self.space_key or name == self.space_key:
                        print(f"     匹配到 space key: {key}")
                        return key

                # 模糊匹配（包含关系）
                for s in results:
                    key = s.get('key', '')
                    name = s.get('name', '')
                    if self.space_key.lower() in key.lower() or self.space_key.lower() in name.lower():
                        print(f"     模糊匹配到 space key: {key}")
                        return key

            print(f"     警告: 未找到匹配的空间，将使用原始值: {self.space_key}")
            return self.space_key
        except Exception as e:
            print(f"     获取空间列表失败: {e}")
            return self.space_key

    def load(self) -> List[Document]:
        """从 Confluence 加载文档"""
        docs = []

        client = self._get_client()

        # 按空间加载
        if self.space_key:
            try:
                print(f"  >>> 正在从空间 [{self.space_key}] 加载...")

                # 获取正确的 space key
                actual_key = self._get_space_key(client)
                if not actual_key:
                    print(f"  ⚠️ 无法确定 space key")
                    return docs

                # 获取空间内所有页面
                pages = client.get_all_pages_from_space(
                    space=actual_key,
                    start=0,
                    limit=100
                )

                print(f"     获取到 {len(pages)} 个页面")

                for page in pages:
                    try:
                        page_id = page.get('id')
                        # 使用 view 格式获取渲染后的 HTML（更干净）
                        page_detail = client.get_page_by_id(page_id, expand='body.view')

                        content = page_detail.get('body', {}).get('view', {}).get('value', '')
                        title = page_detail.get('title', 'untitled')

                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"confluence:{actual_key}:{title}",
                                "page_id": page_id,
                                "title": title,
                                "url": page_detail.get('_links', {}).get('webui', '')
                            }
                        )
                        docs.append(doc)
                    except Exception as e:
                        print(f"    ⚠️ 加载页面失败: {e}")

                print(f"  ✓ 从空间 {actual_key} 加载了 {len(docs)} 个页面")
            except Exception as e:
                print(f"  ⚠️ 加载空间 {self.space_key} 失败: {e}")

        # 按页面 ID 加载
        if self.page_ids:
            for page_id in self.page_ids:
                try:
                    print(f"  >>> 正在加载页面 {page_id}...")
                    # 使用 view 格式获取渲染后的 HTML
                    page_detail = client.get_page_by_id(page_id, expand='body.view')

                    content = page_detail.get('body', {}).get('view', {}).get('value', '')
                    title = page_detail.get('title', 'untitled')

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"confluence:page:{page_id}",
                            "page_id": page_id,
                            "title": title,
                            "url": page_detail.get('_links', {}).get('webui', '')
                        }
                    )
                    docs.append(doc)
                    print(f"  ✓ 加载页面 {page_id}: {title}")
                except Exception as e:
                    print(f"  ⚠️ 加载页面 {page_id} 失败: {e}")

        return docs

    def get_source_name(self) -> str:
        return "confluence"
