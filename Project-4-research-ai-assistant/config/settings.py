import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
ARXIV_DATASET_NAME = "Cornell-University/arxiv"
RAW_JSON_FILE = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json"
PROCESSED_CSV_FILE = PROCESSED_DATA_DIR / "arxiv_cs_filtered.csv"
MAX_PAPERS_TO_PROCESS = 50000  # Limit to avoid massive processing time locally, adjustable

# Computer Science Categories
CS_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.DS": "Data Structures and Algorithms",
    "cs.RO": "Robotics"
}

# Retrieval Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 10
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434")
