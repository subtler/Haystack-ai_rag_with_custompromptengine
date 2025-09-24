# src/pipelines.py

import json
from typing import List, Dict, Any

from haystack import component, Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.embedders.amazon_bedrock import (
    AmazonBedrockTextEmbedder,
    AmazonBedrockDocumentEmbedder,
)
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from . import config

@component
class CustomPromptEngine:
    """
    An enhanced prompt engine that builds a structured prompt asking for a JSON output
    and includes rich metadata for better context.
    """
    @component.output_types(prompt=str)
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Builds and returns the final prompt with metadata filtering."""

        system_prompt = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a data-centric AI assistant for a CRM system. Your primary function is to provide structured, factual answers based *only* on the provided context.
Your response MUST be a single, valid JSON object and nothing else.

**JSON Schema:**
{
  "answer": "string | A concise answer to the user's question, synthesized from the context. If the answer is not in the context, this field MUST say 'I could not find a relevant answer in the provided documents.'",
  "references": "[string] | An array of the titles of the documents used to formulate the answer. This array MUST be empty if no relevant documents were found."
}
<|eot_id|>
"""
        context_block = ""
        for doc in documents:
            # --- METADATA FILTERING ---
            # Include rich metadata in the context for the LLM
            context_block += f"""
---
Document Title: {doc.meta.get('title', 'N/A')}
Category: {doc.meta.get('category', 'N/A')}
Folder: {doc.meta.get('folder', 'N/A')}
Tags: {', '.join(doc.meta.get('tags', [])) if doc.meta.get('tags') else 'N/A'}
Content: {doc.content}
"""
        final_prompt = f"""
{system_prompt}
<|start_header_id|>user<|end_header_id|>
**Context:**
{context_block}
---
**Question:** {query}

**JSON Response:**
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
        return {"prompt": final_prompt}


@component
class JsonOutputParser:
    """
    A component that safely parses a JSON string from the LLM into a dictionary.
    """
    @component.output_types(result=Dict[str, Any])
    def run(self, replies: List[str]):
        try:
            json_string = replies[0].strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]

            parsed_json = json.loads(json_string)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"JSON parsing failed: {e}\nRaw reply: {replies[0] if replies else 'None'}")
            parsed_json = {
                "answer": "Sorry, the response from the language model was not in the expected JSON format.",
                "references": []
            }
        return {"result": parsed_json}


def build_rag_pipeline(document_store: PineconeDocumentStore) -> Pipeline:
    """Builds the RAG pipeline with the enhanced custom prompt engine."""

    text_embedder = AmazonBedrockTextEmbedder(model=config.EMBEDDING_MODEL_ID)
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=3)
    prompt_engine = CustomPromptEngine()
    llm = AmazonBedrockGenerator(model=config.GENERATOR_MODEL_ID)
    parser = JsonOutputParser()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_engine", prompt_engine)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.add_component("parser", parser)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_engine.documents")
    rag_pipeline.connect("prompt_engine.prompt", "llm.prompt")
    rag_pipeline.connect("llm.replies", "parser.replies")

    print("✅ RAG pipeline with metadata-enhanced prompt built successfully.")
    return rag_pipeline


def build_indexing_pipeline(document_store: PineconeDocumentStore) -> Pipeline:
    """Builds a pipeline to embed and write documents to Pinecone."""
    writer = DocumentWriter(document_store=document_store, policy="overwrite")
    pipeline = Pipeline()
    pipeline.add_component("embedder", AmazonBedrockDocumentEmbedder(model=config.EMBEDDING_MODEL_ID))
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline

