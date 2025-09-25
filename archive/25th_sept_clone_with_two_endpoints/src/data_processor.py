# src/data_processor.py

import json
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from haystack import Document

def process_json_data(data: List[Dict[str, Any]]) -> List[Document]:
    """
    Processes a list of dictionaries (from a JSON object) and prepares Haystack Documents.
    This is the core processing logic.
    """
    documents = []
    for category in data:
        for folder in category.get("folders", []):
            for article in folder.get("articles", []):
                content_to_process = article.get('description_text', '')
                title = article.get('title', '')
                
                raw_content = f"{title}. {content_to_process}"
                clean_content = BeautifulSoup(raw_content, "html.parser").get_text().strip()
                
                if clean_content and article.get("id"):
                    metadata = {
                        "article_id": article.get("id"),
                        "category": category.get("category_name"),
                        "folder": folder.get("folder_name"),
                        "title": title,
                        "tags": article.get("tags", []),
                    }
                    
                    doc_id = str(article.get("id"))
                    
                    documents.append(
                        Document(
                            id=doc_id,
                            content=clean_content,
                            meta=metadata
                        )
                    )
    print(f"Successfully processed {len(documents)} documents from JSON data.")
    return documents

def load_and_process_data(file_path: Path) -> List[Document]:
    """
    Loads a JSON file from a given path and processes it.
    This function is a convenient wrapper for file-based operations.
    """
    print(f"Loading data from file: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return process_json_data(json_data)
