# ui.py

import streamlit as st
import re
from typing import Tuple


# Import core components from your src package
from src import config
from src.pipelines import build_rag_pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

@st.cache_resource
def load_rag_pipeline():
    """Loads the RAG pipeline and document store."""
    document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
    return build_rag_pipeline(document_store)

def parse_response(response_text: str) -> Tuple[str, str]:
    """Parses the LLM's text output to separate the answer and references."""
    answer_match = re.search(r"ANSWER:(.*?)REFERENCES:", response_text, re.DOTALL)
    references_match = re.search(r"REFERENCES:(.*)", response_text, re.DOTALL)

    answer = answer_match.group(1).strip() if answer_match else response_text
    references_text = references_match.group(1).strip() if references_match else "No specific documents were cited."
    
    # Format references as a bulleted list
    references_list = [ref.strip() for ref in references_text.split(',') if ref.strip()]
    formatted_references = ""
    if references_list:
        for ref in references_list:
            formatted_references += f"- *{ref}*\n"
    else:
        formatted_references = "*No specific documents were cited.*\n"

    return answer, formatted_response

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
                # The pipeline output is now simpler
                result = rag_pipeline.run(
                    data={
                        "text_embedder": {"text": prompt},
                        "prompt_engine": {"query": prompt}
                    }
                )
                
                raw_answer = result["llm"]["replies"][0]
                
                # --- KEY CHANGE: Manually parse the text response ---
                answer = raw_answer
                references = "Could not parse references from the response." # Default value

                # Simple parsing logic
                if "ANSWER:" in raw_answer and "REFERENCES:" in raw_answer:
                    parts = raw_answer.split("REFERENCES:")
                    answer_part = parts[0].replace("ANSWER:", "").strip()
                    references_part = parts[1].strip()
                    
                    answer = answer_part
                    references = "\n\n---\n**References:**\n"
                    ref_list = [ref.strip() for ref in references_part.split(',') if ref.strip()]
                    if ref_list:
                         for ref in ref_list:
                            references += f"- *{ref}*\n"
                    else:
                        references = "\n\n---\n*No specific documents were cited.*\n"
                
                formatted_response = f"{answer}{references}"
                        
                st.markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})

            except Exception as e:
                print(f"An error occurred: {e}") 
                error_msg = "Sorry, I encountered an error. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})   