import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import VECTOR_DB_PATH
from models.embeddings import EmbeddingModel

class ChromaVectorStore:
    def __init__(self, collection_name="medquad"):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        self.embedding_model = EmbeddingModel()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_records(self, ids, texts, metadatas):
        """
        Add records in batches to avoid memory issues.
        """
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=metadatas[i:i+batch_size]
            )

    def search(self, query: str, top_k: int = 10, filter_metadata: dict = None):
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filter_metadata if filter_metadata else None
        )
        
        return results
