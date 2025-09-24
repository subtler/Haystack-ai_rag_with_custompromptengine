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
    print("Initializing RAG pipeline...")
    initialize_pinecone_index()
    
    document_store = PineconeDocumentStore(
        index=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION
    )

    if document_store.count_documents() == 0:
        print("Document store is empty. Starting the indexing process...")
        docs_to_index = load_and_process_data(config.DATA_FILE_PATH)
        indexing_pipeline = build_indexing_pipeline(document_store)
        indexing_pipeline.run({"embedder": {"documents": docs_to_index}})
        print(f"✅ Indexing complete. {len(docs_to_index)} documents written to Pinecone.")
    else:
        print("Documents already indexed.")

    rag_pipeline = build_rag_pipeline(document_store)
    print("\n✅ RAG pipeline is ready to answer questions.")
    print("="*50)

    while True:
        query = input("Ask a question (or type 'quit' to exit): ")
        
        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not query.strip():
            continue
        
        # --- KEY CHANGE 1: Correctly pass the query to the prompt_engine ---
        result = rag_pipeline.run(
            data={
                "text_embedder": {"text": query},
                "prompt_engine": {"query": query}
            }
        )

        # --- KEY CHANGE 2: Access the structured JSON from the parser ---
        parsed_result = result["parser"]["result"]
        answer = parsed_result.get("answer", "No answer found.")
        references = parsed_result.get("references", [])

        # --- KEY CHANGE 3: Print the response from the JSON object ---
        print("\n" + "="*50)
        print("✅ ANSWER:")
        print(answer)
        
        print("\n" + "-"*50)
        print("📚 REFERENCES:")
        if not references:
            print("No references cited.")
        else:
            for i, ref in enumerate(references):
                print(f"  [{i+1}] {ref}")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()

