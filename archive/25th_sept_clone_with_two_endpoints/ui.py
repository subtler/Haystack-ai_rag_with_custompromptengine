#ui.py
import streamlit as st

# Import core components from your src package
from src import config
from src.pipelines import build_rag_pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

@st.cache_resource
def load_rag_pipeline():
    """Loads the RAG pipeline and document store."""
    document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
    return build_rag_pipeline(document_store)

st.set_page_config(page_title="CRM Assistant", layout="wide")
st.title("🤖 CRM Support Assistant")
st.markdown("Ask me anything about our CRM! I will find the answer in our official documentation.")

rag_pipeline = load_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I change my account details?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for answers..."):
            try:
                # --- KEY CHANGE 1: Correctly pass the query to the prompt_engine ---
                result = rag_pipeline.run(
                    data={
                        "text_embedder": {"text": prompt},
                        "prompt_engine": {"query": prompt}
                    }
                )
                
                # --- KEY CHANGE 2: Access the structured JSON output from the parser ---
                parsed_result = result["parser"]["result"]
                answer = parsed_result.get("answer", "No answer found.")
                references = parsed_result.get("references", [])
                
                # --- KEY CHANGE 3: Format the response directly from the JSON ---
                formatted_response = answer
                if references:
                    formatted_response += "\n\n---\n**References:**\n"
                    for ref in references:
                        formatted_response += f"- *{ref}*\n"
                
                st.markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})

            except Exception as e:
                print(f"An error occurred in the RAG pipeline: {e}")
                error_msg = "Sorry, I encountered an error. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

