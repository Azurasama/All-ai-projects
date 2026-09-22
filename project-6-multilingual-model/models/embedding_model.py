from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

class EmbeddingModel:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        # This will download the model locally on first run and use it for subsequent runs
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def get_embeddings(self):
        return self.embeddings
