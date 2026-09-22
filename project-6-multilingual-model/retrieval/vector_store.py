import chromadb
from typing import List, Dict, Any
import uuid
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
from models.embedding_model import EmbeddingModel

class VectorStore:
    def __init__(self):
        # Initialize the ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.embedding_model = EmbeddingModel().get_embeddings()
        
        # Get or create the collection for storing chat history
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for embeddings
        )

    def add_message(self, session_id: str, message: str, role: str, detected_language: str, intent: str = "Unknown"):
        """
        Embeds and stores a message in the vector database.
        """
        message_id = str(uuid.uuid4())
        
        # Generate embedding using sentence-transformers
        embedding = self.embedding_model.embed_query(message)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[message_id],
            embeddings=[embedding],
            documents=[message],
            metadatas=[{
                "session_id": session_id,
                "role": role,
                "language": detected_language,
                "intent": intent
            }]
        )

    def search_similar_messages(self, query: str, session_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most semantically similar messages from the same session.
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"session_id": session_id} # Only search within the current conversation
        )
        
        retrieved_messages = []
        if results['documents'] and len(results['documents']) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            for doc, meta in zip(docs, metas):
                retrieved_messages.append({
                    "content": doc,
                    "metadata": meta
                })
                
        return retrieved_messages
    
    def clear_session(self, session_id: str):
        """
        Clears all stored messages for a given session.
        """
        try:
            self.collection.delete(where={"session_id": session_id})
        except Exception as e:
            print(f"Error clearing session: {e}")
