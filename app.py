# app.py

import io
import json
import logging
from typing import List

# --- Core Libraries ---
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse

# --- Haystack & Pinecone Imports ---
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# --- Modular Imports from `src` package ---
from src import config
from src.data_processor import process_json_data, load_and_process_data 
from src.pipelines import build_rag_pipeline, build_indexing_pipeline
from src.schemas import RAGResponse, QueryRequest, UpdateResponse

# ======================================================================================
# 1. INITIAL CONFIGURATION & SETUP
# ======================================================================================
logging.basicConfig(level=logging.INFO)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="CRM Support RAG API",
    description="An API for querying a RAG system and updating its knowledge base.",
    version="1.0.0",
)

# --- Global Haystack Components ---
document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
RAG_PIPELINE = build_rag_pipeline(document_store)
logging.info("✅ Global RAG pipeline loaded and ready.")


# ======================================================================================
# 2. API ENDPOINTS
# ======================================================================================

@app.post("/query", response_model=RAGResponse, summary="Ask a question to the RAG system")
async def ask_question(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        logging.info(f"Running query: {request.query}")
        result = RAG_PIPELINE.run(
            data={
                "text_embedder": {"text": request.query},
                "prompt_engine": {"query": request.query}
            }
        )
        final_output = result["parser"]["result"]
        return final_output
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        error_response = {
            "answer": "Sorry, an internal error occurred while processing your request.",
            "references": []
        }
        return JSONResponse(status_code=500, content=error_response)


@app.post("/update-articles", response_model=UpdateResponse, summary="Update articles in Pinecone from a JSON file")
async def update_articles_from_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .json file.")
    try:
        file_content = await file.read()
        json_data = json.loads(file_content.decode('utf-8'))
        docs_to_index = process_json_data(json_data)
        
        if not docs_to_index:
            raise HTTPException(status_code=400, detail="No valid documents found in the uploaded file.")

        logging.info(f"Starting indexing for {len(docs_to_index)} documents...")
        indexing_pipeline = build_indexing_pipeline(document_store)
        indexing_pipeline.run({"embedder": {"documents": docs_to_index}})
        return {"message": f"Successfully indexed {len(docs_to_index)} documents.", "documents_processed": len(docs_to_index)}

    # --- KEY FIX: More specific error handling ---
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly so FastAPI can handle them
        raise http_exc
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in the uploaded file.")
    except Exception as e:
        # Catch any other unexpected errors as a 500
        logging.error(f"Error during article update: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# ======================================================================================
# 3. RUN THE APPLICATION (for local development)
# ======================================================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

