🤖 Production-Grade RAG System for CRM Support
This repository contains a robust, production-ready Retrieval-Augmented Generation (RAG) application designed to answer questions about a CRM system. It is built using the Haystack-AI framework and leverages a modern stack including Pinecone for vector storage and AWS Bedrock for state-of-the-art language models.

The system is accessible via a user-friendly Streamlit web interface and a command-line interface. It includes a comprehensive, automated evaluation suite to continuously measure performance and security.

✨ Core Technologies
Technology

Purpose

Haystack-AI

The core framework for building and orchestrating the RAG pipeline.

Pinecone

High-performance serverless vector database for document retrieval.

AWS Bedrock

Provides access to powerful foundation models (e.g., Llama 3) for embedding and generation.

Streamlit

Powers the interactive and user-friendly web interface.

Pydantic

Enforces a strict data contract for reliable, structured LLM outputs.

FastAPI

(Optional) The app.py provides an API-first interface for the system.

🏛️ System Architecture
The application follows a modular, API-first design. The core logic is built around a Haystack pipeline that processes user queries and generates validated, structured responses.

The data flow is as follows:

A User Query is received from either the Streamlit UI or the command-line interface.

The query is sent to the Haystack RAG Pipeline.

Text Embedder: The query is converted into a vector embedding.

Retriever: The vector is used to search Pinecone for the most relevant document chunks.

Prompt Engine: The retrieved documents and the original query are inserted into a highly-engineered prompt template. This prompt commands the LLM to follow strict rules, including a security mandate and a JSON-only output format.

LLM (AWS Bedrock): The prompt is sent to a powerful language model (e.g., Llama 3) to generate a response.

Pydantic Parser: The LLM's response is immediately validated against a strict Pydantic schema. This ensures the output is always a clean, predictable JSON object.

Structured Response: The final, validated JSON is returned to the user interface for display.

🧠 Understanding Haystack-AI
Since Haystack is a new and powerful framework, it's important to understand its core concepts.

What is Haystack?
Haystack is an open-source framework that helps you build applications with Large Language Models. Instead of writing complex, monolithic code, Haystack allows you to build Pipelines from modular Components.

Core Concepts: Pipelines and Components
Think of Haystack like building with LEGOs.

Components: These are the individual LEGO bricks. Each component has one specific job (e.g., retrieve documents, generate text, embed a query). Haystack provides many pre-built components, and you can easily create your own, like our CustomPromptEngine.

Pipelines: This is what you build with the LEGO bricks. A pipeline defines how data flows from one component to the next. You simply add components to a pipeline and then connect() them together.

Our RAG Pipeline Explained
Our src/pipelines.py script defines the main RAG pipeline. Here is a breakdown of the components we use:

Component

Role in the Pipeline

AmazonBedrockTextEmbedder

The Translator: Converts the user's text query into a vector (a list of numbers) that Pinecone can understand.

PineconeEmbeddingRetriever

The Librarian: Takes the vector and efficiently searches through millions of documents in Pinecone to find the most relevant ones.

CustomPromptEngine

The Master Chef: This is our custom component. It takes the raw ingredients (the user's query and the retrieved documents) and assembles them into a highly-structured, advanced prompt according to our recipe.

AmazonBedrockGenerator

The Brain: This is the LLM. It receives the complex prompt from the Prompt Engine and generates the answer.

ValidatedJsonOutputParser

The Quality Inspector: This is our Pydantic-powered component. It checks the LLM's raw output to ensure it's a perfectly formed JSON object that matches our schema. This prevents errors and ensures reliability.

This modular approach makes the system easy to understand, debug, and upgrade. For example, to switch to a new LLM, we would only need to change the AmazonBedrockGenerator component.

📁 Project Structure
/
├── data/
│   ├── evaluation_dataset.json  # The "Golden Set" for testing
│   └── final_4dcrm_articles_clean.json # The raw knowledge base
├── src/
│   ├── __init__.py
│   ├── config.py              # Central configuration (API keys, model IDs)
│   ├── data_processor.py      # Cleans and prepares data for indexing
│   ├── pipelines.py           # Defines the core Haystack RAG pipeline
│   └── schemas.py             # Pydantic schema for the final LLM output
├── utils/
│   ├── run_evaluation.py      # Automated script to test pipeline quality & security
│   ├── run_indexing.py        # Script to load data into Pinecone
│   └── test_access_checker.py # Utility to verify AWS Bedrock model access
├── .env                       # Stores secret keys (DO NOT COMMIT)
├── README.md                  # This file
├── requirements.txt           # Project dependencies
├── run.py                     # Command-line interface for the RAG system
└── ui.py                      # Streamlit web interface

🚀 Setup and Installation
1. Prerequisites
Python 3.9+

An active AWS account with Bedrock access enabled for the desired models.

A Pinecone account and API key.

2. Clone the Repository
git clone <your-repo-url>
cd <your-repo-name>

3. Configure Environment Variables
Create a file named .env in the root of the project and add your secret keys.

PINECONE_API_KEY="your-pinecone-api-key"

# Configure your AWS credentials here or via system-level profiles
AWS_ACCESS_KEY_ID="your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
AWS_DEFAULT_REGION="us-east-1" # Or your preferred region

4. Install Dependencies
pip install -r requirements.txt

🛠️ How to Run
1. Index Your Data
Before you can ask questions, you must load your knowledge base into Pinecone. Run the indexing script from the project root:

python utils/run_indexing.py

This will process the file specified in src/config.py and embed it in your Pinecone index.

2. Run the Streamlit Web UI
To interact with the chatbot through a graphical interface:

streamlit run ui.py

Open your web browser to the local URL provided by Streamlit.

3. Run the Command-Line Interface
For a terminal-based interaction:

python run.py

4. Run the Automated Evaluation Suite
To test the pipeline's quality and security against the golden dataset:

python utils/run_evaluation.py

This will run all the test cases and print a detailed performance report to the console.

🧠 Prompt Engineering Strategy
The reliability of this system comes from the advanced prompt engineering used in src/pipelines.py. The CustomPromptEngine applies several key principles:

Strict JSON Enforcement: The prompt commands the LLM to only respond with a valid JSON object, which is then validated by our Pydantic parser.

Chain-of-Thought (CoT): The prompt includes "Deliberation Steps" that force the LLM to think methodically before answering, improving accuracy.

Security Mandate: An explicit security directive is included to make the prompt more resilient to injection attacks.

XML-Style Delimiters: Using tags like <context> and <question> provides a clear structure that modern LLMs are highly adept at understanding.

This multi-pronged approach is what enables the pipeline to pass the rigorous tests in our evaluation suite.