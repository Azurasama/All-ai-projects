import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import CHROMA_DB_DIR

class VectorStore:
    def __init__(self):
        # Allow running in Streamlit, disable telemetry
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "arxiv_cs_papers"
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # Fallback if it doesn't exist, though prepare_data should run first
            self.collection = self.client.create_collection(name=self.collection_name)

    def search_by_vector(self, query_vector: list[float], top_k: int = 10, where_filter: dict = None):
        """
        Searches ChromaDB using a query vector.
        Optionally filters by metadata.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        return results

    def get_paper_by_id(self, paper_id: str):
        """Fetches a specific paper by its arXiv ID."""
        results = self.collection.get(
            ids=[paper_id],
            include=['documents', 'metadatas']
        )
        if not results['ids']:
            return None
        return {
            'id': results['ids'][0],
            'document': results['documents'][0],
            'metadata': results['metadatas'][0]
        }

# Singleton instance
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
