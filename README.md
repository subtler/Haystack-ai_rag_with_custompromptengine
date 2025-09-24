# 🤖 Production-Grade RAG System for CRM Support

This repository contains a robust, production-ready Retrieval-Augmented Generation (RAG) application designed to answer questions about a CRM system. It is built using the **Haystack-AI** framework and leverages a modern stack including **Pinecone** for vector storage and **AWS Bedrock** for state-of-the-art language models.

The system is accessible via a **Streamlit web interface** and a **command-line interface**. It includes a comprehensive, automated evaluation suite to continuously measure performance and security.

---

## ✨ Core Technologies

| Technology  | Purpose                                                              |
| ----------- | -------------------------------------------------------------------- |
| Haystack-AI | The core framework for building and orchestrating the RAG pipeline   |
| Pinecone    | High-performance serverless vector database for document retrieval   |
| AWS Bedrock | Provides access to powerful foundation models (e.g., Llama 3)        |
| Streamlit   | Powers the interactive and user-friendly web interface               |
| Pydantic    | Enforces a strict data contract for reliable, structured LLM outputs |
| FastAPI     | *(Optional)* Provides an API-first interface for the system          |

---

## 🏛️ System Architecture

The application follows a modular, API-first design. The core logic is built around a **Haystack pipeline** that processes user queries and generates validated, structured responses.

**Data Flow:**

1. A **User Query** is received via the Streamlit UI or CLI.
2. The query is sent to the **Haystack RAG Pipeline**:

   * **Text Embedder**: Converts query to a vector embedding
   * **Retriever**: Searches Pinecone using the vector
   * **Prompt Engine**: Crafts a structured prompt from query + docs
   * **LLM (AWS Bedrock)**: Generates a response from the prompt
   * **Pydantic Parser**: Validates output against schema
3. A clean, predictable **JSON** response is returned to the user interface.

---

## 🧠 Understanding Haystack-AI

**What is Haystack?**

Haystack is an open-source framework that helps you build applications with Large Language Models using modular **Pipelines** and **Components**.

* **Components** = building blocks (e.g., embedder, retriever, generator)
* **Pipelines** = a flowchart that connects components step-by-step

---

## 🔁 Our RAG Pipeline Components (from `src/pipelines.py`)

| Component                  | Role/Metaphor                                                      |
| -------------------------- | ------------------------------------------------------------------ |
| AmazonBedrockTextEmbedder  | *The Translator*: Converts user query to vector embeddings         |
| PineconeEmbeddingRetriever | *The Librarian*: Retrieves matching documents from Pinecone        |
| CustomPromptEngine         | *The Master Chef*: Crafts the prompt with query + documents        |
| AmazonBedrockGenerator     | *The Brain*: Generates the LLM response from the prompt            |
| ValidatedJsonOutputParser  | *The Inspector*: Validates output against a strict Pydantic schema |

This modular design makes the system easy to understand, debug, and extend.

---

## 📁 Project Structure

```shell
/
├── data/
│   ├── evaluation_dataset.json        # Golden Set for testing
│   └── final_4dcrm_articles_clean.json # Raw knowledge base
├── src/
│   ├── __init__.py
│   ├── config.py                      # API keys, model IDs
│   ├── data_processor.py              # Cleans & prepares data
│   ├── pipelines.py                   # RAG pipeline definition
│   └── schemas.py                     # Pydantic output schema
├── utils/
│   ├── run_evaluation.py              # Evaluation/test suite
│   ├── run_indexing.py                # Loads data to Pinecone
│   └── test_access_checker.py         # Checks Bedrock access
├── .env                               # Secret keys (DO NOT COMMIT)
├── README.md
├── requirements.txt                  # Python dependencies
├── run.py                             # CLI interface
└── ui.py                              # Streamlit frontend
```

---

## 🚀 Setup and Installation

### 1. Prerequisites

* Python 3.9+
* AWS account with Bedrock access
* Pinecone account and API key

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 3. Configure Environment Variables

Create a `.env` file with:

```env
PINECONE_API_KEY="your-pinecone-api-key"
AWS_ACCESS_KEY_ID="your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
AWS_DEFAULT_REGION="us-east-1"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### 1. Index Your Data

```bash
python utils/run_indexing.py
```

This will embed and index your knowledge base.

### 2. Run Streamlit Web UI

```bash
streamlit run ui.py
```

### 3. Run CLI Interface

```bash
python run.py
```

### 4. Run Evaluation Suite

```bash
python utils/run_evaluation.py
```

---

## 🧠 Prompt Engineering Strategy

**What makes this system reliable?** Our `CustomPromptEngine` applies:

* **Strict JSON Enforcement**: Prompt explicitly instructs output as valid JSON
* **Chain-of-Thought (CoT)**: Includes deliberate thinking steps
* **Security Mandate**: Prompt includes injection protection
* **XML-style Delimiters**: Uses tags like `<context>` and `<question>`

This strategy ensures:

* Higher accuracy
* Safer execution
* Seamless downstream parsing with Pydantic

---

✅ With modular architecture, rigorous prompt engineering, and a complete test suite, this RAG system is ready for production use in CRM support and beyond.
