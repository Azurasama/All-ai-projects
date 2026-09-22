import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import EMBEDDING_MODEL

class EmbeddingModel:
    """Wrapper for the SentenceTransformer embedding model."""
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
        
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

# Singleton instance
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
