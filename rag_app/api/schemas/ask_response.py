from pydantic import BaseModel
from typing import Optional, Dict, Any
class AskResponse(BaseModel):
    # query:str
    answer:str
    context:str
    evaluation: Optional[Dict[str, Any]] = None