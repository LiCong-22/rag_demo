from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import engine, get_rag_config, set_rag_config

app = FastAPI(title="AutoElec RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    thinking: str = ""
    sources: list[str]

class ConfigRequest(BaseModel):
    enable_hyde: bool | None = None
    enable_query_expansion: bool | None = None

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    try:
        result = engine.query(request.question)
        # 确保 answer 是字符串
        answer = result["answer"]
        if hasattr(answer, 'content'):
            answer = answer.content
        answer = str(answer)
        thinking = result.get("thinking", "")
        return QueryResponse(answer=answer, thinking=thinking, sources=result["sources"])
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ 查询错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/config")
async def get_config():
    """获取当前 RAG 配置"""
    return get_rag_config()

@app.post("/config")
async def update_config(request: ConfigRequest):
    """修改 RAG 配置"""
    try:
        new_config = set_rag_config(
            enable_hyde=request.enable_hyde,
            enable_query_expansion=request.enable_query_expansion
        )
        return new_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)