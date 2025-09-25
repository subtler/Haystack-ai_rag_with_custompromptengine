# 🤖 Production-Grade RAG System for CRM Support

This repository contains a robust, production-ready Retrieval-Augmented Generation (RAG) application designed to answer questions about a CRM system. It is built using the **Haystack-AI** framework and leverages a modern stack including **Pinecone** for vector storage and **AWS Bedrock** for state-of-the-art language models.

The system is accessible via a **Streamlit web interface** and a **command-line interface**. It includes a comprehensive, automated evaluation suite to continuously measure performance and security.

---

## ✨ Core Technologies

| Technology           | Purpose                                                              |
| -------------------- | -------------------------------------------------------------------- |
| Haystack-AI          | The core framework for building and orchestrating the RAG pipeline   |
| Pinecone             | High-performance serverless vector database for document retrieval   |
| AWS Bedrock          | Provides access to powerful foundation models (e.g., Llama 3)        |
| Streamlit            | Powers the interactive and user-friendly web interface               |
| Pydantic             | Enforces a strict data contract for reliable, structured LLM outputs |
| FastAPI *(Optional)* | `app.py` provides an API-first interface for the system              |

---

## 🏛️ System Architecture

The application follows a **modular**, **API-first** design. The core logic is built around a **Haystack pipeline** that processes user queries and generates validated, structured responses.

### Data flow:

1. **User Query** is received from either the Streamlit UI or the command-line interface.
2. The query is sent to the **Haystack RAG Pipeline**:

   * **Text Embedder**: The query is converted into a vector embedding.
   * **Retriever**: The vector is used to search Pinecone for the most relevant document chunks.
   * **Prompt Engine**: Retrieved documents and the original query are inserted into a highly-engineered prompt template. This prompt commands the LLM to follow strict rules, including a security mandate and a JSON-only output format.
   * **LLM (AWS Bedrock)**: The prompt is sent to a powerful language model (e.g., Llama 3) to generate a response.
   * **Pydantic Parser**: The LLM's response is validated against a strict Pydantic schema to ensure it’s always clean, predictable JSON.
3. **Structured Response**: The validated JSON is returned to the user interface for display.

---

## 🧠 Understanding Haystack-AI

### What is Haystack?

Haystack is an open-source framework that helps you build applications with Large Language Models. Instead of writing complex, monolithic code, Haystack allows you to build **Pipelines** from modular **Components**.

* **Components**: Like LEGO bricks. Each has one specific job (e.g., retrieve documents, generate text, embed a query). You can also build custom components like our `CustomPromptEngine`.
* **Pipelines**: The structure built with LEGO bricks. A pipeline defines how data flows from one component to another.

---

## 🔁 Our RAG Pipeline Explained (from `src/pipelines.py`)

| Component                  | Role in the Pipeline                                                    |
| -------------------------- | ----------------------------------------------------------------------- |
| AmazonBedrockTextEmbedder  | The Translator: Converts text queries into vectors for Pinecone         |
| PineconeEmbeddingRetriever | The Librarian: Searches Pinecone for the most relevant documents        |
| CustomPromptEngine         | The Master Chef: Crafts advanced structured prompts from queries + docs |
| AmazonBedrockGenerator     | The Brain: Generates answers from the LLM                               |
| ValidatedJsonOutputParser  | The Quality Inspector: Ensures outputs are valid JSON via Pydantic      |

This modular approach makes the system easy to debug, swap components, or upgrade.

---

## 📁 Project Structure

```bash
/
├── data/
│   ├── evaluation_dataset.json
│   └── final_4dcrm_articles_clean.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processor.py
│   ├── pipelines.py
│   └── schemas.py
├── utils/
│   ├── run_evaluation.py
│   ├── run_indexing.py
│   └── test_access_checker.py
├── .env
├── README.md
├── requirements.txt
├── run.py
└── ui.py
```

---

## 🚀 Setup and Installation

### 1. Prerequisites

* Python 3.9+
* An active AWS account with Bedrock access enabled for the desired models.
* A Pinecone account and API key.

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 3. Configure Environment Variables

Create a file named `.env` in the root of the project and add your secret keys:

```env
PINECONE_API_KEY="your-pinecone-api-key"

# Configure your AWS credentials here or via system-level profiles
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

Before you can ask questions, you must load your knowledge base into Pinecone:

```bash
python utils/run_indexing.py
```

### 2. Run the Streamlit Web UI

To interact with the chatbot through a graphical interface:

```bash
streamlit run ui.py
```

### 3. Run the Command-Line Interface

For a terminal-based interaction:

```bash
python run.py
```

### 4. Run the Automated Evaluation Suite

To test the pipeline's quality and security against the golden dataset:

```bash
python utils/run_evaluation.py
```

---

## 🧠 Prompt Engineering Strategy

The reliability of this system comes from the advanced prompt engineering used in `src/pipelines.py`:

* **Strict JSON Enforcement**: The prompt commands the LLM to only respond with a valid JSON object, which is then validated by our Pydantic parser.
* **Chain-of-Thought (CoT)**: The prompt includes "Deliberation Steps" that force the LLM to think methodically before answering, improving accuracy.
* **Security Mandate**: An explicit security directive is included to make the prompt more resilient to injection attacks.
* **XML-Style Delimiters**: Using tags like `<context>` and `<question>` provides a clear structure that modern LLMs are highly adept at understanding.

---
