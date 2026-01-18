from fastapi import APIRouter, UploadFile, File
import uuid
import os

from services.upload_service import process_pdf

router = APIRouter()

# @router.post("/upload")
# def upload_pdf(file: UploadFile = File(...)):
#     session_id = str(uuid.uuid4())
#     save_path = f"storage/uploads/{session_id}.pdf"

#     os.makedirs("storage/uploads", exist_ok=True)

#     with open(save_path, "wb") as f:
#         f.write(file.file.read())

#     process_pdf(save_path, session_id)

#     return {"session_id": session_id}

@router.post("/upload")
def upload_pdf(file: UploadFile, session_id: str):
    path = f"uploads/{session_id}/{file.filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(file.file.read())

    router.ingest(path, session_id)

    return {"status": "uploaded"}