from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import EMBEDDING_MODEL_NAME

class EmbeddingModel:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Returns embeddings as a list of lists of floats
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
