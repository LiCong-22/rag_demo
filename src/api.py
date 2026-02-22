from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import engine

app = FastAPI(title="AutoElec RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    try:
        result = engine.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)