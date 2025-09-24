# src/schemas.py

from typing import List
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    """Defines the strict, validated JSON output schema for our RAG pipeline."""
    answer: str = Field(description="The final, synthesized answer to the user's question.")
    references: List[str] = Field(description="A list of document titles used as sources for the answer.")
