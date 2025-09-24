# run.py

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone, ServerlessSpec

# Import from our source package
from src import config
from src.data_processor import load_and_process_data
from src.pipelines import build_indexing_pipeline, build_rag_pipeline

def initialize_pinecone_index():
    """Checks for and creates the Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{config.PINECONE_INDEX_NAME}' not found. Creating a new serverless index...")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Serverless index '{config.PINECONE_INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{config.PINECONE_INDEX_NAME}' already exists. Skipping creation.")

def main():
    """Main function to orchestrate the RAG pipeline setup and execution."""
    # --- 1. INITIALIZATION ---
    print("Initializing RAG pipeline...")
    initialize_pinecone_index()
    
    document_store = PineconeDocumentStore(
        index=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION
    )

    # --- 2. INDEXING (if needed) ---
    if document_store.count_documents() == 0:
        print("Document store is empty. Starting the indexing process...")
        docs_to_index = load_and_process_data(config.DATA_FILE_PATH)
        indexing_pipeline = build_indexing_pipeline(document_store)
        indexing_pipeline.run({"embedder": {"documents": docs_to_index}})
        print(f"✅ Indexing complete. {len(docs_to_index)} documents written to Pinecone.")
    else:
        print("Documents already indexed.")

    # --- 3. BUILD QUERY PIPELINE ---
    rag_pipeline = build_rag_pipeline(document_store)
    print("\n✅ RAG pipeline is ready to answer questions.")
    print("="*50)

    # --- 4. INTERACTIVE CHAT LOOP (IMPROVED OUTPUT) ---
    while True:
        query = input("Ask a question (or type 'quit' to exit): ")
        
        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not query.strip():
            continue
        
        # Run the pipeline with the user's query
        result = rag_pipeline.run(
            data={
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            },
            include_outputs_from=["retriever"]
        )

        # --- Structured Output ---
        print("\n" + "="*50)

        # Print the Generated Answer
        print("✅ ANSWER:")
        print(result["llm"]["replies"][0].strip())
        
        print("\n" + "-"*50)

        # Print the Retrieved Sources
        print("📚 SOURCES:")
        retrieved_docs = result["retriever"]["documents"]
        if not retrieved_docs:
            print("No sources found.")
        else:
            for i, doc in enumerate(retrieved_docs):
                # doc.score shows how relevant the document was (higher is better)
                score = round(doc.score * 100, 2)
                print(f"  [{i+1}] {doc.meta.get('title', 'N/A')} (Relevance: {score}%)")
                print(f"      - Category: {doc.meta.get('category', 'N/A')}")
                print(f"      - Folder: {doc.meta.get('folder', 'N/A')}")

        print("="*50 + "\n")


if __name__ == "__main__":
    main()