# from fastapi import FastAPI
# from pydantic import BaseModel
# from services.rag_service import RAGService
# import uuid
# import os
# from fastapi import FastAPI, UploadFile, File
# from schemas.ask_response import AskResponse
# app = FastAPI(
#     title="Hybrid RAG API",
#     version="1.0.0"
# )

# rag = RAGService()   # loaded once at startup
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/upload")
# def upload_pdf(file: UploadFile = File(...)):
#     session_id = str(uuid.uuid4())

#     file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
#     with open(file_path, "wb") as f:
#         f.write(file.file.read())

#     rag.ingest(file_path, session_id)

#     return {
#         "message": "PDF uploaded and indexed",
#         "session_id": session_id,
#         "file": file.filename
#     }
# class QueryRequest(BaseModel):
#     question: str
#     session_id: str
    

# class QueryResponse(BaseModel):
#     answer: str

# # @app.post("/ask", response_model=QueryResponse)
# # def ask_rag(req: QueryRequest):
# #     answer = rag.ask(req.question,req.session_id)
# #     return {"answer": answer}
# @app.post("/ask", response_model=AskResponse)
# def ask_rag(req: AskRequest):
#     result = rag.ask(req.question, req.session_id)

#     return result


# @app.get("/")
# def root():
#     return {
#         "status": "RAG API running",
#         "docs": "/docs"
#     }
from fastapi import FastAPI
from pydantic import BaseModel
from services.rag_service import RAGService
import uuid
import os
from fastapi import FastAPI, UploadFile, File
from api.schemas.ask_response import AskResponse
app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0"
)

rag = RAGService()   # loaded once at startup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    rag.ingest(file_path, session_id)

    return {
        "message": "PDF uploaded and indexed",
        "session_id": session_id,
        "file": file.filename
    }
class QueryRequest(BaseModel):
    question: str
    session_id: str
    

class QueryResponse(BaseModel):
    answer: str

# @app.post("/ask", response_model=QueryResponse)
# def ask_rag(req: QueryRequest):
#     answer = rag.ask(req.question,req.session_id)
#     return {"answer": answer}
@app.post("/ask", response_model=AskResponse)
def ask_rag(req: QueryRequest):
    result = rag.ask(req.question, req.session_id)

    return result


@app.get("/")
def root():
    return {
        "status": "RAG API running",
        "docs": "/docs"
    }