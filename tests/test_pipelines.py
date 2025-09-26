# tests/test_pipelines.py

import pytest
from haystack import Document

# This allows the test script to find the 'src' package
from src.pipelines import CustomPromptEngine, ValidatedJsonOutputParser

# --- Tests for CustomPromptEngine ---

def test_custom_prompt_engine_happy_path():
    """
    Tests that the prompt engine correctly constructs a prompt with all necessary parts.
    """
    prompt_engine = CustomPromptEngine()
    query = "What are the payment methods?"
    documents = [
        Document(content="We accept credit cards.", meta={"title": "Payments Doc", "category": "Billing", "folder": "General", "tags": ["payment"]})
    ]
    result = prompt_engine.run(query=query, documents=documents)
    prompt = result["prompt"]
    assert "<|system|>" in prompt
    assert query in prompt
    assert "We accept credit cards." in prompt
    assert "<title>Payments Doc</title>" in prompt

def test_custom_prompt_engine_no_documents():
    """
    Tests that the prompt engine handles an empty list of documents gracefully.
    """
    prompt_engine = CustomPromptEngine()
    query = "What about obscure features?"
    documents = []
    result = prompt_engine.run(query=query, documents=documents)
    prompt = result["prompt"]
    assert query in prompt
    assert "<context>\n</context>" in prompt
    assert "<document>" not in prompt

def test_custom_prompt_engine_with_special_characters():
    """
    ADVANCED TEST: Ensures the prompt handles unicode and special characters correctly.
    """
    prompt_engine = CustomPromptEngine()
    query = "Cómo se dice 'payment' en español? 🤔"
    documents = [Document(content="La respuesta es 'pago'.", meta={"title": "Traducción Español"})]
    result = prompt_engine.run(query=query, documents=documents)
    prompt = result["prompt"]
    assert query in prompt
    assert "La respuesta es 'pago'." in prompt
    assert "Traducción Español" in prompt


# --- Tests for ValidatedJsonOutputParser ---

def test_validated_json_parser_valid_json():
    """
    Tests that the parser correctly handles a perfect JSON string.
    """
    parser = ValidatedJsonOutputParser()
    valid_json_string = ['{"answer": "This is the answer.", "references": ["Doc 1"]}']
    result = parser.run(replies=valid_json_string)
    output = result["result"]
    assert output["answer"] == "This is the answer."
    assert output["references"] == ["Doc 1"]

def test_validated_json_parser_invalid_json():
    """
    Tests that the parser returns a structured error for malformed JSON.
    """
    parser = ValidatedJsonOutputParser()
    invalid_json_string = ['{"answer": "Forgot a quote", "references": ["Doc 1}']
    result = parser.run(replies=invalid_json_string)
    output = result["result"]
    assert "Error: The model's response was not valid or did not match the required schema" in output["answer"]

def test_validated_json_parser_with_extra_text():
    """
    Tests that the parser can extract and validate JSON even if the LLM adds extra text.
    """
    parser = ValidatedJsonOutputParser()
    chatty_llm_output = [
        'Sure, here is the JSON you requested:\n'
    ]

