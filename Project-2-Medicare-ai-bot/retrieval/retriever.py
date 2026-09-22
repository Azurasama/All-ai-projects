import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.vector_store import ChromaVectorStore
from config.settings import TOP_K_SEMANTIC, RELEVANCE_THRESHOLD
from utils.logging import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self):
        self.vector_store = ChromaVectorStore()

    def retrieve(self, query: str, entities: dict, question_type: str, top_k=TOP_K_SEMANTIC):
        """
        Retrieves relevant documents using semantic search, optionally boosting by entities or question_type.
        """
        # We can construct a metadata filter for ChromaDB if we want strict filtering,
        # but for medical queries, strict filtering on "focus" might be too harsh if NER misses something.
        # We'll just do semantic search and optionally re-rank or filter post-retrieval.
        
        results = self.vector_store.search(query, top_k=top_k)
        
        if not results['ids'] or not results['ids'][0]:
            return []

        retrieved_docs = []
        
        # Results structure from ChromaDB:
        # results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]
        # Distances are cosine distance (1 - cosine_similarity). Lower is better.
        
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        for i in range(len(ids)):
            # Convert cosine distance to similarity
            similarity = 1.0 - distances[i]
            
            # Simple metadata boosting
            metadata = metadatas[i]
            boost = 0.0
            
            if question_type and question_type != "general" and metadata.get("question_type") == question_type:
                boost += 0.1
                
            # If the focus matches extracted entities, boost
            focus = metadata.get("focus", "").lower()
            if focus:
                for ent_list in entities.values():
                    for ent in ent_list:
                        if ent.lower() in focus or focus in ent.lower():
                            boost += 0.2
                            break

            final_score = similarity + boost
            
            if final_score >= RELEVANCE_THRESHOLD:
                retrieved_docs.append({
                    "id": ids[i],
                    "metadata": metadata,
                    "similarity": similarity,
                    "final_score": final_score
                })
        
        # Sort by final score
        retrieved_docs.sort(key=lambda x: x["final_score"], reverse=True)
        return retrieved_docs
