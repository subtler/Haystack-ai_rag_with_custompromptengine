# src/data_processor.py

import json
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from haystack import Document

def load_and_process_data(file_path: Path) -> List[Document]:
    """
    Loads JSON data, cleans HTML content, and prepares Haystack Documents
    with consistent IDs to enable overwriting.
    """
    documents = []
    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category in data:
        for folder in category.get("folders", []):
            for article in folder.get("articles", []):
                
                # --- MODIFICATION 1: Explicitly use 'description_text' ---
                # This ensures we always use the clean, plain-text version.
                content_to_process = article.get('description_text', '')
                title = article.get('title', '')
                
                raw_content = f"{title}. {content_to_process}"
                clean_content = BeautifulSoup(raw_content, "html.parser").get_text().strip()
                
                # Ensure the document has an ID and content before processing
                if clean_content and article.get("id"):
                    metadata = {
                        "article_id": article.get("id"),
                        "category": category.get("category_name"),
                        "folder": folder.get("folder_name"),
                        "title": title,
                        "tags": article.get("tags", []),
                    }
                    
                    # --- MODIFICATION 2: Assign a consistent ID to the Document ---
                    # This is crucial for overwriting existing documents in Pinecone.
                    doc_id = str(article.get("id"))
                    
                    documents.append(
                        Document(
                            id=doc_id,
                            content=clean_content,
                            meta=metadata
                        )
                    )
    
    print(f"Successfully loaded and processed {len(documents)} documents.")
    return documents