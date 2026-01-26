from fastapi import APIRouter
from pydantic import BaseModel
from services.rag_service import RAGService
from api.schemas.ask_schema import AskRequest
# from llm-evaluation-platform.evaluation.runner import evaluate_rag

router = APIRouter()

class AskRequest(BaseModel):
    session_id: str
    question: str

@router.post("/ask")
def ask(req: AskRequest):
    rag = RAGService(req.session_id)
    # answer = rag.ask(req.question,req.session_id)
    # return {"answer": answer}
    result = rag.ask(req.question, req.session_id)
    # return {"answer": result["answer"]}
    return {
        # "query": req.question,          # ✅ REQUIRED
        "answer": result["answer"],
        "context": result["context"],
        "evaluation": result.get("evaluation")
    }

    
    
