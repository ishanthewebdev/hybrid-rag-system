from fastapi import APIRouter
from pydantic import BaseModel
from services.rag_service import RAGService

router = APIRouter()

class AskRequest(BaseModel):
    session_id: str
    question: str

@router.post("/ask")
def ask(req: AskRequest):
    rag = RAGService(req.session_id)
    answer = rag.ask(req.question,req.session_id)
    return {"answer": answer}
