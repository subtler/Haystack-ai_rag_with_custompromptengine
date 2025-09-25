# src/schemas.py

from typing import List, Optional
from pydantic import BaseModel

# --- 1. CORE RAG RESPONSE SCHEMA ---
# This defines the structured JSON output from our RAG pipeline.
class RAGResponse(BaseModel):
    """The final, validated output of the RAG pipeline."""
    answer: str
    references: List[str]


# --- 2. API REQUEST/RESPONSE SCHEMAS ---
# These models are used by FastAPI for request validation and response formatting.

class QueryRequest(BaseModel):
    """The expected input for the /query endpoint."""
    query: str

class UpdateResponse(BaseModel):
    """The success response for the /update-articles endpoint."""
    message: str
    documents_processed: int


# --- 3. UPLOADED DATA VALIDATION SCHEMAS ---
# These models validate the structure of the JSON file uploaded to /update-articles.
# This ensures data integrity before it ever reaches the indexing pipeline.

class Article(BaseModel):
    id: int
    title: str
    description_text: Optional[str] = ""
    tags: Optional[List[str]] = []

class Folder(BaseModel):
    folder_name: str
    articles: List[Article]

class Category(BaseModel):
    category_name: str
    folders: List[Folder]

