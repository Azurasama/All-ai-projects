import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # LLM Settings
    # Options: "groq", "huggingface", "gemini"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    
    # Model name depending on the provider
    GROQ_MODEL = "llama-3.1-8b-instant" # Or llama3-70b-8192
    HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    GEMINI_MODEL = "gemini-1.5-flash"
    
    # Embedding Settings
    # A fast, lightweight multilingual model that runs locally on CPU
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Vector DB Settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "chat_history"
    
    # Chatbot Context Settings
    MAX_CONTEXT_MESSAGES = 10
    SIMILARITY_TOP_K = 3

settings = Settings()
