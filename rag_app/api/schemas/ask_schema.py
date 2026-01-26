from pydantic import BaseModel
from typing import Optional
class AskRequest(BaseModel):
    question: str
    session_id: str
    ground_truth: Optional[str] = None
