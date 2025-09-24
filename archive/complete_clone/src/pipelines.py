
# src/pipelines.py

from typing import List, Dict, Any

# --- CORRECTED IMPORT ---
from haystack import component, Document, Pipeline
# ------------------------

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
    """A custom component to build a prompt that asks for a delimited text output."""

    @component.output_types(prompt=str)
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Builds and returns the final prompt."""
        
        system_prompt = """
You are an expert AI assistant. Your task is to answer the user's question based ONLY on the context provided below.

**Rules:**
1.  Base your answer solely on the information within the provided documents. Do not use external knowledge.
2.  If the context does not contain the answer, state: "I could not find a relevant answer in the provided documents."
3.  Structure your response using the exact format below, with "ANSWER:" and "REFERENCES:" as the delimiters.

**Example Format:**
---
ANSWER: This is the answer to the user's question based on the context.
REFERENCES: Document Title 1, Document Title 2
"""
        context_block = ""
        for i, doc in enumerate(documents):
            context_block += f"**Document [{i+1}] Title:** {doc.meta.get('title', 'N/A')}\n"
            context_block += f"**Content:**\n{doc.content}\n---\n"

        final_prompt = f"""
{system_prompt}

**Live Request:**
---
**Context:**
{context_block}
**Question:** {query}
"""
        return {"prompt": final_prompt}


def build_rag_pipeline(document_store: PineconeDocumentStore) -> Pipeline:
    """Builds the RAG pipeline using the custom prompt engine."""
    
    text_embedder = AmazonBedrockTextEmbedder(model=config.EMBEDDING_MODEL_ID)
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=3)
    prompt_engine = CustomPromptEngine()
    llm = AmazonBedrockGenerator(model=config.GENERATOR_MODEL_ID)
    
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_engine", prompt_engine)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_engine.documents")
    rag_pipeline.connect("prompt_engine.prompt", "llm.prompt")
    
    print("✅ RAG querying pipeline built successfully with prompt-based formatting.")
    return rag_pipeline

def build_indexing_pipeline(document_store: PineconeDocumentStore) -> Pipeline:
    """Builds a pipeline to embed and write documents to Pinecone."""
    writer = DocumentWriter(document_store=document_store, policy="overwrite")
    pipeline = Pipeline()
    pipeline.add_component("embedder", AmazonBedrockDocumentEmbedder(model=config.EMBEDDING_MODEL_ID))
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline
