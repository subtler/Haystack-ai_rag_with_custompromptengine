# tests/test_data_processor.py

import pytest
from haystack import Document

# This allows the test script to find the 'src' package
from src.data_processor import process_json_data

def test_process_json_data_happy_path():
    """
    Tests the core logic of the data processor with a perfect, simple input.
    """
    sample_data = [
        {
            "category_name": "Test Category",
            "folders": [
                {
                    "folder_name": "Test Folder",
                    "articles": [
                        {
                            "id": 98765,
                            "title": "Test Article Title",
                            "description_text": "This is the clean description.",
                            "tags": ["tag1", "tag2"]
                        }
                    ]
                }
            ]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    doc = processed_documents[0]
    assert isinstance(doc, Document)
    assert doc.content == "Test Article Title. This is the clean description."
    assert doc.id == "98765"
    assert doc.meta["title"] == "Test Article Title"
    assert doc.meta["category"] == "Test Category"
    assert doc.meta["tags"] == ["tag1", "tag2"]

def test_process_json_data_missing_fields():
    """
    Tests how the function handles data with missing optional keys.
    """
    sample_data = [
        {
            "category_name": "Another Category",
            "folders": [
                {
                    "folder_name": "Another Folder",
                    "articles": [
                        {
                            "id": 12345,
                            "title": "Title Only Article"
                        }
                    ]
                }
            ]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    doc = processed_documents[0]
    assert doc.content == "Title Only Article."
    assert doc.meta["tags"] == []

def test_process_json_data_with_empty_input():
    """
    Tests that the function handles an empty list as input without crashing.
    """
    processed_documents = process_json_data([])
    assert isinstance(processed_documents, list)
    assert len(processed_documents) == 0

def test_process_json_data_skips_articles_without_id():
    """
    Tests that the function correctly filters out articles that are missing a required 'id'.
    """
    sample_data = [
        {
            "category_name": "Test Category",
            "folders": [{"articles": [{"id": 111, "title": "Valid Article"}, {"title": "Article Missing ID"}]}]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    assert processed_documents[0].id == "111"

def test_process_json_data_handles_empty_collections():
    """
    Tests that the function is robust to empty 'folders' or 'articles' lists.
    """
    sample_data = [
        {"category_name": "Category with Empty Folder", "folders": [{"folder_name": "Empty Articles Folder", "articles": []}]},
        {"category_name": "Category with No Folders", "folders": []}
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 0

def test_process_json_data_skips_articles_with_no_content():
    """
    Tests that articles with an ID but no title or description are skipped.
    """
    sample_data = [
        {
            "category_name": "Test Category",
            "folders": [{"articles": [{"id": 777}, {"id": 888, "title": "Valid"}]}]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    assert processed_documents[0].id == "888"

def test_process_json_data_handles_html_in_description():
    """
    Tests that HTML tags are properly stripped from the description.
    """
    sample_data = [
        {
            "category_name": "HTML Test",
            "folders": [{"articles": [{"id": 654, "title": "HTML Article", "description_text": "<p>This is <b>bold</b> text.</p>"}]}]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    assert processed_documents[0].content == "HTML Article. This is bold text."

def test_process_json_data_handles_missing_title():
    """
    Tests that an article with a description but no title is still processed.
    """
    sample_data = [
        {
            "category_name": "Missing Title Test",
            "folders": [{"articles": [{"id": 321, "description_text": "Content without a title."}]}]
        }
    ]
    processed_documents = process_json_data(sample_data)
    assert len(processed_documents) == 1
    doc = processed_documents[0]
    assert doc.content == ". Content without a title."
    # --- KEY FIX: The test now correctly asserts for an empty string ---
    assert doc.meta["title"] == ""

def test_process_json_data_with_complex_structure():
    """
    Tests the function with a mix of valid, invalid, and empty data structures.
    """
    sample_data = [
        { # Category 1: Valid
            "category_name": "Sales",
            "folders": [
                {"folder_name": "Leads", "articles": [{"id": 101, "title": "How to handle a lead"}]},
                {"folder_name": "Deals", "articles": [{"id": 102, "title": "Closing a deal"}]}
            ]
        },
        { # Category 2: Mixed valid and invalid
            "category_name": "Support",
            "folders": [
                {"folder_name": "Tickets", "articles": [{"id": 201, "title": "Resolving a ticket"}, {"title": "Ticket without ID"}]},
                {"folder_name": "FAQs", "articles": []} # Empty articles
            ]
        },
        { # Category 3: Empty
             "category_name": "Marketing",
             "folders": []
        }
    ]
    processed_documents = process_json_data(sample_data)
    # Should find 3 valid articles: 101, 102, and 201
    assert len(processed_documents) == 3
    doc_ids = sorted([doc.id for doc in processed_documents])
    assert doc_ids == ["101", "102", "201"]

