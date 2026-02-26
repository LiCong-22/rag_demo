# RAG Knowledge Base System

A RAG (Retrieval-Augmented Generation) knowledge base system for automotive electronics software Q&A, built with LangChain, Milvus, and local/API LLM.

## Features

- **Multiple LLM Support**: Local model (Qwen2.5-7B) or API (OpenAI/Anthropic/MiniMax)
- **Vector Search**: Milvus vector database for efficient document retrieval
- **Web UI**: Streamlit-based chat interface
- **REST API**: FastAPI backend service

## Quick Start

### 1. Start Milvus

```bash
docker-compose up -d
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure LLM

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Edit `src/config.py` or `.env`:
- `LLM_TYPE`: Choose `local`, `openai`, or `anthropic`
- Set your API key accordingly

### 4. Ingest Documents

```bash
python -m src.ingest
```

### 5. Run Services

```bash
# Terminal 1: Start API
python -m src.api

# Terminal 2: Start UI
python -m src.app
```

Access the web UI at http://localhost:8501

## Architecture

```
Streamlit UI → FastAPI → RAG Engine → Milvus → Documents
```

## Project Structure

| File | Description |
|------|-------------|
| `src/config.py` | Configuration management |
| `src/ingest.py` | Document loading and vectorization |
| `src/engine.py` | RAG engine with retrieval and LLM generation |
| `src/api.py` | FastAPI REST API |
| `src/app.py` | Streamlit web UI |

## Supported Document Formats

- `.md` (Markdown)
- `.txt` (Plain text)

---

# RAG 知识库系统

基于 LangChain、Milvus 和本地/API 大语言模型的汽车电子软件问答知识库系统。

## 功能特性

- **多 LLM 支持**：本地模型 (Qwen2.5-7B) 或 API (OpenAI/Anthropic/MiniMax)
- **向量检索**：Milvus 向量数据库，高效文档检索
- **Web 界面**：基于 Streamlit 的聊天界面
- **REST API**：FastAPI 后端服务

## 快速开始

### 1. 启动 Milvus

```bash
docker-compose up -d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLM

复制 `.env.example` 到 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `src/config.py` 或 `.env`：
- `LLM_TYPE`：选择 `local`、`openai` 或 `anthropic`
- 设置对应的 API Key

### 4. 导入文档

```bash
python -m src.ingest
```

### 5. 启动服务

```bash
# 终端 1：启动 API
python -m src.api

# 终端 2：启动 UI
streamlit run src\app.py
```

访问 http://localhost:8501

## 系统架构

```
Streamlit UI → FastAPI → RAG Engine → Milvus → 文档数据
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `src/config.py` | 配置管理 |
| `src/ingest.py` | 文档加载与向量化 |
| `src/engine.py` | RAG 引擎：向量检索 + LLM 生成 |
| `src/api.py` | FastAPI REST 接口 |
| `src/app.py` | Streamlit Web 界面 |

## 支持的文档格式

- `.md` (Markdown)
- `.txt` (纯文本)

