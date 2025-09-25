# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at the project root
load_dotenv()

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "crm-articles-rag"

# --- AWS Bedrock Configuration ---
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATOR_MODEL_ID = "us.meta.llama3-2-11b-instruct-v1:0"
#GENERATOR_MODEL_ID = "meta.llama3-8b-instruct-v1:0"

EMBEDDING_DIMENSION = 1024

# --- Data Configuration ---
# Assumes the script is run from the project root (crm_rag_app/)
DATA_FILE_PATH = Path("data/4dcrm_articles_demo.json")