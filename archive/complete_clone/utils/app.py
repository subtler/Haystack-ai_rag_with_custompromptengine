# app.py
# A single-file prototype combining a Haystack RAG pipeline, FastAPI endpoints, and a Streamlit UI.
#
# --- HOW TO RUN ---
# 1. Install all dependencies:
#    pip install "fastapi[all]" "streamlit" "haystack-ai" "pinecone-haystack" "amazon-bedrock-haystack" "python-dotenv" "beautifulsoup4" "boto3"
#
# 2. To run the Streamlit Chat App:
#    streamlit run app.py
#
# 3. To run the FastAPI Server:
#    uvicorn app:api --reload --port 8000
#
# ------------------

import os
import json
import logging
from pathlib import Path
from typing import List, Optional

# --- Core Libraries ---
import uvicorn
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- Haystack & Pinecone Imports ---
from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.embedders.amazon_bedrock import (
    AmazonBedrockTextEmbedder,
    AmazonBedrockDocumentEmbedder,
)
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from pinecone import Pinecone, ServerlessSpec

# ======================================================================================
# 1. INITIAL CONFIGURATION & SETUP
# ======================================================================================
logging.basicConfig(level=logging.WARNING)
load_dotenv()

# --- Load Configuration from .env ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "crm-articles-rag"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATOR_MODEL_ID = "us.meta.llama3-2-11b-instruct-v1:0"
EMBEDDING_DIMENSION = 1024
DATA_FILE_PATH = Path("data/4dcrm_articles_demo.json")

# ======================================================================================
# 2. ADVANCED PROMPT ENGINEERING
# ======================================================================================
# This refined prompt template provides a clear persona, strict rules, and a structured
# format to ensure the LLM provides high-quality, referenced answers.

ADVANCED_PROMPT_TEMPLATE = """
You are an expert AI assistant for our CRM system. Your primary goal is to provide accurate and helpful answers based *only* on the context provided below.

**Rules:**
1.  Examine the context carefully before answering.
2.  Your answer must be based solely on the information within the provided documents. Do not use any external knowledge.
3.  If the context does not contain the answer to the question, you must explicitly state: "I could not find a relevant answer in the provided documents."
4.  After providing the answer, you must cite your sources in a "References" section, listing the title of each document used.

**Context:**
{% for doc in documents %}
    **Document Title:** {{ doc.meta.get('title', 'N/A') }}
    **Content:**
    {{ doc.content }}
    ---
{% endfor %}

**Question:** {{ query }}

**Answer:**
"""

# ======================================================================================
# 3. CORE HAYSTACK & PINECONE LOGIC
# ======================================================================================
# This section contains all the core functions for initializing the vector store,
# processing data, and building the Haystack pipelines.

@st.cache_resource
def get_document_store():
    """Initializes and returns a PineconeDocumentStore instance."""
    # Initialize Pinecone Index if it doesn't exist
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created successfully.")
    
    return PineconeDocumentStore(index=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIMENSION)

def process_json_data(json_data: List[dict]) -> List[Document]:
    """Loads JSON data, cleans it, and prepares Haystack Documents with consistent IDs."""
    documents = []
    for category in json_data:
        for folder in category.get("folders", []):
            for article in folder.get("articles", []):
                raw_content = f"{article.get('title', '')}. {article.get('description_text', '')}"
                clean_content = BeautifulSoup(raw_content, "html.parser").get_text().strip()

                if clean_content and article.get("id"):
                    metadata = {
                        "article_id": article.get("id"),
                        "category": category.get("category_name"),
                        "folder": folder.get("folder_name"),
                        "title": article.get("title"),
                    }
                    documents.append(
                        Document(
                            id=str(article.get("id")),
                            content=clean_content,
                            meta=metadata
                        )
                    )
    return documents

def build_indexing_pipeline(doc_store: PineconeDocumentStore) -> Pipeline:
    """Builds a pipeline to embed and write documents to Pinecone."""
    writer = DocumentWriter(document_store=doc_store, policy="overwrite")
    pipeline = Pipeline()
    pipeline.add_component("embedder", AmazonBedrockDocumentEmbedder(model=EMBEDDING_MODEL_ID))
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline

@st.cache_resource
def build_rag_pipeline(_doc_store: PineconeDocumentStore) -> Pipeline:
    """Builds the RAG pipeline for querying."""
    text_embedder = AmazonBedrockTextEmbedder(model=EMBEDDING_MODEL_ID)
    retriever = PineconeEmbeddingRetriever(document_store=_doc_store, top_k=3)
    prompt_builder = PromptBuilder(template=ADVANCED_PROMPT_TEMPLATE)
    llm = AmazonBedrockGenerator(model=GENERATOR_MODEL_ID)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    return pipeline

# Initialize components for FastAPI
document_store = get_document_store()
rag_pipeline = build_rag_pipeline(document_store)


# ======================================================================================
# 4. FASTAPI APPLICATION
# ======================================================================================
# This section defines the API endpoints using FastAPI.

api = FastAPI(
    title="RAG API",
    description="API for querying and updating a RAG system with Haystack.",
)

# --- Pydantic Models for API validation ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    references: List[str]

class Article(BaseModel):
    id: int
    title: str
    description_text: Optional[str] = None

class Folder(BaseModel):
    folder_name: str
    articles: List[Article]

class Category(BaseModel):
    category_name: str
    folders: List[Folder]

@api.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Endpoint for asking questions. It runs the query through the RAG pipeline.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = rag_pipeline.run({
            "text_embedder": {"text": request.question},
            "prompt_builder": {"query": request.question}
        })
        
        answer = result["llm"]["replies"][0]
        # A simple way to extract references mentioned in the text
        references = [doc.meta.get("title", "Unknown") for doc in result["retriever"]["documents"]]
        
        return QueryResponse(answer=answer, references=list(set(references)))

    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

@api.post("/update-articles", status_code=202)
def update_articles(data: List[Category]):
    """
    Endpoint for updating articles in Pinecone. It processes the provided JSON
    and runs the indexing pipeline.
    """
    try:
        # Pydantic automatically validates the incoming data structure.
        # We need to convert it back to a dict for our processing function.
        json_data = [category.model_dump() for category in data]
        
        docs_to_index = process_json_data(json_data)
        
        if not docs_to_index:
            raise HTTPException(status_code=400, detail="No valid documents to index.")

        indexing_pipeline = build_indexing_pipeline(document_store)
        indexing_pipeline.run({"embedder": {"documents": docs_to_index}})
        
        return {"message": f"Indexing started for {len(docs_to_index)} documents."}

    except Exception as e:
        logging.error(f"Error during article update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update articles.")

# ======================================================================================
# 5. STREAMLIT APPLICATION
# ======================================================================================
# This section defines the Streamlit chat interface. It is only active when
# the script is run with `streamlit run app.py`.

def run_streamlit_app():
    st.set_page_config(page_title="CRM Assistant", layout="wide")
    st.title("🤖 CRM Support Assistant")
    st.markdown("Ask a question about our CRM system, and I'll find the answer in our knowledge base.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("How can I change my account details?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Reuse the RAG pipeline
                    result = rag_pipeline.run({
                        "text_embedder": {"text": prompt},
                        "prompt_builder": {"query": prompt}
                    })
                    
                    answer = result["llm"]["replies"][0]
                    references = [doc.meta.get("title", "Unknown") for doc in result["retriever"]["documents"]]
                    
                    # Format the response with references
                    formatted_response = f"{answer}\n\n---\n**References:**\n"
                    for ref in set(references):
                        formatted_response += f"- *{ref}*\n"
                        
                    st.markdown(formatted_response)
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})

                except Exception as e:
                    # --- KEY CHANGE: Print the actual error to the terminal ---
                    print(f"An error occurred: {e}") 
                    
                    error_message = "Sorry, I encountered an error. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    # This block allows the script to be run as a Streamlit app
    run_streamlit_app()