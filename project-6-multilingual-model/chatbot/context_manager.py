from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
from retrieval.vector_store import VectorStore

class ContextManager:
    def __init__(self):
        self.vector_store = VectorStore()
        # In-memory storage for immediate recent history (per session)
        self.recent_history: Dict[str, List[Dict]] = {}
        
    def add_message(self, session_id: str, role: str, content: str, language: str = "Unknown", intent: str = "Unknown"):
        """
        Adds a message to both the recent history memory and the vector store.
        """
        # Initialize session if not exists
        if session_id not in self.recent_history:
            self.recent_history[session_id] = []
            
        message_data = {
            "role": role,
            "content": content,
            "language": language,
            "intent": intent
        }
        
        # Add to recent history (sliding window)
        self.recent_history[session_id].append(message_data)
        if len(self.recent_history[session_id]) > settings.MAX_CONTEXT_MESSAGES:
            self.recent_history[session_id].pop(0)
            
        # Add to vector store for semantic retrieval
        self.vector_store.add_message(
            session_id=session_id,
            message=content,
            role=role,
            detected_language=language,
            intent=intent
        )
        
    def get_recent_history(self, session_id: str, k: int = 5) -> List[Dict]:
        """
        Returns the last k messages from the conversation.
        """
        if session_id not in self.recent_history:
            return []
        return self.recent_history[session_id][-k:]
        
    def get_semantic_context(self, session_id: str, query: str) -> str:
        """
        Retrieves semantically similar messages to provide broader context,
        especially useful for resolving cross-lingual coreferences.
        """
        results = self.vector_store.search_similar_messages(
            query=query, 
            session_id=session_id, 
            k=settings.SIMILARITY_TOP_K
        )
        
        if not results:
            return ""
            
        context_str = "Relevant past conversation context:\n"
        for res in results:
            role = res['metadata']['role']
            lang = res['metadata']['language']
            content = res['content']
            context_str += f"- [{lang}] {role}: {content}\n"
            
        return context_str
        
    def clear_context(self, session_id: str):
        if session_id in self.recent_history:
            self.recent_history[session_id] = []
        self.vector_store.clear_session(session_id)
