# src/schemas.py

from typing import List
from pydantic import BaseModel

class RAGResponse(BaseModel):
    """Defines the structured JSON output for our RAG pipeline."""
    answer: str
    references: List[str]# src/schemas.py

from typing import List
from pydantic import BaseModel

class RAGResponse(BaseModel):
    """Defines the structured JSON output for our RAG pipeline."""
    answer: str
    references: List[str]