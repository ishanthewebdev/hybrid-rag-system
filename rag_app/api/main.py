from fastapi import FastAPI
from pydantic import BaseModel
from services.rag_service import RAGService

app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0"
)

rag = RAGService()   # loaded once at startup

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_rag(req: QueryRequest):
    answer = rag.ask(req.question)
    return {"answer": answer}


@app.get("/")
def root():
    return {
        "status": "RAG API running",
        "docs": "/docs"
    }