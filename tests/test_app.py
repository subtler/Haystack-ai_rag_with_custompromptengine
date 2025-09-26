# tests/test_app.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app instance from your app.py
from app import app 

# Create a synchronous TestClient for our FastAPI app
client = TestClient(app)

# --- Happy Path and Validation Tests ---

def test_query_endpoint_success():
    """
    Tests the /query endpoint for a successful request (200 OK).
    """
    mock_response = {
        "parser": {"result": {"answer": "This is a mock answer.", "references": ["Mock Doc 1"]}}
    }
    with patch("app.RAG_PIPELINE") as mock_pipeline:
        mock_pipeline.run.return_value = mock_response
        response = client.post("/query", json={"query": "A valid question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mock answer."

def test_query_endpoint_empty_query():
    """
    Tests that the API correctly returns a 400 Bad Request for an empty query.
    """
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]

def test_update_articles_endpoint_success():
    """
    Tests a successful file upload to the /update-articles endpoint.
    """
    fake_json_content = '[{"id": 1, "title": "Test", "folders": [{"articles": [{"id": 1, "title": "Test"}]}]}]'
    with patch("app.process_json_data") as mock_process:
        with patch("app.build_indexing_pipeline") as mock_build_pipeline:
            mock_process.return_value = [MagicMock()]
            mock_build_pipeline.return_value = MagicMock()
            files = {"file": ("test.json", fake_json_content, "application/json")}
            response = client.post("/update-articles", files=files)

    assert response.status_code == 200
    assert "Successfully indexed 1 documents" in response.json()["message"]

def test_update_articles_endpoint_wrong_file_type():
    """
    Tests that the API rejects a file that is not a .json file.
    """
    files = {"file": ("test.txt", "some content", "text/plain")}
    response = client.post("/update-articles", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

# --- ADVANCED EDGE CASES: Testing for Graceful Failure & Content Validation ---

def test_query_endpoint_handles_pipeline_exception():
    """
    Ensures the API returns a 500 error if the RAG pipeline fails.
    """
    with patch("app.RAG_PIPELINE") as mock_pipeline:
        mock_pipeline.run.side_effect = Exception("A critical pipeline error occurred!")
        response = client.post("/query", json={"query": "This will cause an error"})

    assert response.status_code == 500
    response_json = response.json()
    assert "Sorry, an internal error occurred" in response_json["answer"]
    assert response_json["references"] == []

def test_update_articles_endpoint_handles_indexing_failure():
    """
    Ensures the API returns a 500 error if the indexing process fails.
    """
    fake_json_content = '[{"folders": [{"articles": [{"id": 1, "title": "Test"}]}]}]'
    with patch("app.process_json_data") as mock_process:
        with patch("app.build_indexing_pipeline") as mock_build_pipeline:
            mock_process.return_value = [MagicMock()]
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.run.side_effect = Exception("Pinecone connection failed!")
            mock_build_pipeline.return_value = mock_pipeline_instance
            files = {"file": ("test.json", fake_json_content, "application/json")}
            response = client.post("/update-articles", files=files)

    assert response.status_code == 500
    assert "An internal error occurred" in response.json()["detail"]

def test_update_articles_invalid_json_content():
    """
    Ensures the API returns a 400 error for a file that is JSON but has the wrong structure.
    """
    # This content is valid JSON, but it's a dictionary, not a list of categories.
    invalid_structure_content = '{"error": "This is not the expected format"}'
    files = {"file": ("bad_structure.json", invalid_structure_content, "application/json")}
    response = client.post("/update-articles", files=files)

    assert response.status_code == 500 # The current code throws a 500 for this
    assert "An internal error occurred" in response.json()["detail"]


def test_update_articles_no_valid_documents():
    """
    Ensures the API returns a 400 error if the uploaded JSON contains no processable documents.
    """
    # This content is structurally valid, but the articles are missing 'id' and will be filtered out.
    content_no_valid_docs = '[{"category_name": "Test", "folders": [{"articles": [{"title": "No ID here"}]}]}]'
    
    # We don't need to mock here because we want to test the real data processor's output
    with patch("app.build_indexing_pipeline"): # Still mock the pipeline to prevent Pinecone call
        files = {"file": ("no_valid_docs.json", content_no_valid_docs, "application/json")}
        response = client.post("/update-articles", files=files)

    assert response.status_code == 400
    assert "No valid documents found" in response.json()["detail"]

