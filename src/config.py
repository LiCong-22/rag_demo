# src/config.py
import os

# ==================== LLM 类型选择 ====================
# 可选值: "local", "openai", "anthropic"
LLM_TYPE = "anthropic"

# ==================== 本地模型配置 (当 LLM_TYPE = "local" 时使用) ====================
LLM_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

# ==================== OpenAI 配置 (当 LLM_TYPE = "openai" 时使用) ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# ==================== Anthropic 配置 (当 LLM_TYPE = "anthropic" 时使用) ====================
# MiniMax Anthropic 兼容 API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.5")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")

# ==================== Milvus 配置 ====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "auto_elec_knowledge"

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 数据源配置 ====================
# 数据源类型: "local", "confluence", "all"
# - "all": 同时加载本地文件和 Confluence（推荐）
# - "local": 只加载本地文件
# - "confluence": 只加载 Confluence
DATA_SOURCE = "all"

# ==================== 本地文件配置 ====================
DATA_PATH = os.path.join(BASE_DIR, "data")

# ==================== Confluence 配置 ====================
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")

# ==================== 分块配置 ====================
# 子块大小 (段落级，用于精准检索)
CHUNK_SIZE = 400
# 子块重叠
CHUNK_OVERLAP = 50
# 父块大小 (章节级，用于提供完整上下文)
PARENT_CHUNK_SIZE = 800
# 启用父子分块
ENABLE_PARENT_CHILD = True

# ==================== RAG 增强配置 ====================
# HyDE: 生成假设文档辅助检索 (会额外调用一次 LLM，增加延迟)
ENABLE_HYDE = True

# 查询扩展: 用 LLM 生成同义问题增加召回率 (会额外调用一次 LLM)
ENABLE_QUERY_EXPANSION = True

# 查询扩展数量
EXPANSION_COUNT = 3

# 检索数量 (增大以容纳更多检索结果)
RETRIEVAL_K = 8