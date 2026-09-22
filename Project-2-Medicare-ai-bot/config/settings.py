import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "MedQuAD"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "evaluation"

# ChromaDB path
VECTOR_DB_PATH = BASE_DIR / "chroma_db"

# Model configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Use a lightweight medical NER model from scispacy if installed, else fallback to standard
# Note: scispacy requires en_core_sci_sm to be installed explicitly
NER_MODEL_NAME = "en_core_web_sm" 

# Retrieval Settings
TOP_K_SEMANTIC = 10
TOP_K_FINAL = 5
RELEVANCE_THRESHOLD = 0.5

# Create necessary directories
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EVAL_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
