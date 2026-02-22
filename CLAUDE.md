# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 **RAG (Retrieval-Augmented Generation) 知识库系统**，用于汽车电子软件领域的技术问答。系统基于 LangChain 框架构建，结合 Milvus 向量数据库和本地大语言模型。

## 常用命令

```bash
# 启动 Milvus 服务（需要 Docker）
docker-compose up -d

# 文档向量化入库
python -m src.ingest

# 启动 FastAPI 后端服务
python -m src.api

# 启动 Streamlit 前端界面
python -m src.app
```

## 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Streamlit UI  │────▶│  FastAPI 后端   │────▶│   RAG Engine   │
│  (src/app.py)  │     │  (src/api.py)   │     │ (src/engine.py)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │ Milvus 向量库   │
                                              │ + MinIO + etcd  │
                                              └─────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │   文档数据      │
                                              │  (data/*.md)    │
                                              └─────────────────┘
```

## 核心模块

| 文件 | 功能 |
|------|------|
| `src/config.py` | 配置管理（模型路径、Milvus 连接、集合名称） |
| `src/ingest.py` | 文档加载、分块、向量化、入库 |
| `src/engine.py` | RAG 引擎：向量检索 + LLM 生成 |
| `src/api.py` | FastAPI RESTful 接口 |
| `src/app.py` | Streamlit 可视化前端 |

## 配置说明

主要配置在 `src/config.py`：

- `LLM_MODEL_PATH`: 大语言模型路径（默认 Qwen/Qwen2.5-7B-Instruct）
- `EMBEDDING_MODEL_PATH`: Embedding 模型路径（默认 BAAI/bge-m3）
- `MILVUS_URI`: Milvus 连接地址（默认 http://localhost:19530）
- `COLLECTION_NAME`: Milvus 集合名称（默认 auto_elec_knowledge）
- `DATA_PATH`: 知识库文档目录（默认 ./data）

## 开发注意事项

1. **首次使用**：需先运行 `docker-compose up -d` 启动 Milvus 服务
2. **更新知识库**：修改 `data/` 目录下的文档后，重新运行 `python -m src.ingest`
3. **前端依赖后端**：启动 Streamlit UI 前需确保 FastAPI 服务运行在 8000 端口
