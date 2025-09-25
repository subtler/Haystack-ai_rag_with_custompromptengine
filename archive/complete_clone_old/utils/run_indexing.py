# utils/run_indexing.py
# A dedicated script to force the re-embedding and overwriting of all documents.

import sys
import os

# --- KEY CHANGE: Add the project's root directory to the Python path ---
# This allows the script to find the 'src' package when run from the 'utils' folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---------------------------------------------------------------------

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from src import config
from src.data_processor import load_and_process_data
from src.pipelines import build_indexing_pipeline

def main():
    """
    This function initializes the document store, loads and processes data using the
    latest logic in data_processor.py, and runs the indexing pipeline to
    overwrite the documents in Pinecone.
    """
    print("Starting dedicated indexing and overwriting process...")
    
    # Step 1: Initialize the connection to your Pinecone index
    document_store = PineconeDocumentStore(
        index=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION
    )
    
    # Step 2: Load and process data using your updated script
    print("Processing data with the latest logic...")
    docs_to_index = load_and_process_data(config.DATA_FILE_PATH)
    
    # Step 3: Build and run the indexing pipeline
    # The 'overwrite' policy in your pipeline ensures existing documents with the same ID are replaced.
    print("Building indexing pipeline...")
    indexing_pipeline = build_indexing_pipeline(document_store)
    
    print(f"Overwriting {len(docs_to_index)} documents in Pinecone...")
    indexing_pipeline.run({"embedder": {"documents": docs_to_index}})
    
    print("\n✅ Process complete. All documents have been re-embedded and overwritten.")

if __name__ == "__main__":
    main()