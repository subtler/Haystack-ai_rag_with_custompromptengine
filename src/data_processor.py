# src/data_processor.py

import json
from typing import List, Dict, Any
from pathlib import Path

from bs4 import BeautifulSoup
from haystack import Document

def process_json_data(json_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Processes a list of dictionary objects into Haystack Documents.
    """
    documents = []
    for category in json_data:
        for folder in category.get("folders", []):
            for article in folder.get("articles", []):
                title = article.get('title', '')
                description = article.get('description_text', '')
                
                raw_content = f"{title}. {description}"
                clean_content = BeautifulSoup(raw_content, "html.parser").get_text().strip()
                
                # --- KEY FIX 1: Stricter check to filter out empty/useless documents ---
                # An article is only valid if it has an ID and meaningful text content.
                if article.get("id") and clean_content and clean_content != '.':
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
    Loads a JSON file from a path and processes it into Haystack Documents.
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return process_json_data(data)

