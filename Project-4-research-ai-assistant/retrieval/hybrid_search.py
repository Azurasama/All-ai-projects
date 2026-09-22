import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.embeddings import get_embedding_model
from retrieval.vector_store import get_vector_store
from config.settings import TOP_K_RETRIEVAL

def build_where_clause(categories: list = None, year: str = None, author: str = None) -> dict:
    """Builds the metadata filter for ChromaDB."""
    conditions = []
    
    if categories:
        # If multiple categories, we need an $or condition. 
        # But Chroma 'where' supports simple exact match. 
        # Since categories are space-separated strings in our DB, we use $contains
        if len(categories) == 1:
            conditions.append({"categories": {"$contains": categories[0]}})
        else:
            or_conds = [{"categories": {"$contains": c}} for c in categories]
            conditions.append({"$or": or_conds})
            
    if year:
        # update_date is string like YYYY-MM-DD
        conditions.append({"update_date": {"$contains": str(year)}})
        
    if author:
        conditions.append({"authors": {"$contains": author}})
        
    if not conditions:
        return None
        
    if len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

class HybridSearcher:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL, filters: dict = None):
        """
        Executes hybrid search.
        Since ChromaDB doesn't have native BM25 out-of-the-box, we rely on semantic search
        via embeddings, and precise metadata filtering. 
        
        Args:
            query (str): The search query.
            top_k (int): Number of results to return.
            filters (dict): Dictionary with 'categories', 'year', 'author'
        """
        # Parse filters
        where_clause = None
        if filters:
            where_clause = build_where_clause(
                categories=filters.get("categories"),
                year=filters.get("year"),
                author=filters.get("author")
            )
            
        # Get query embedding
        query_vector = self.embedding_model.embed_text(query)
        
        # Search
        raw_results = self.vector_store.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            where_filter=where_clause
        )
        
        # Format results
        formatted_results = []
        if raw_results['ids'] and len(raw_results['ids']) > 0:
            for i in range(len(raw_results['ids'][0])):
                # Calculate a dummy "relevance score" from distance
                # Cosine distance to similarity roughly: 1 - distance
                distance = raw_results['distances'][0][i]
                relevance_score = round(max(0, 1 - distance), 4)
                
                formatted_results.append({
                    "id": raw_results['ids'][0][i],
                    "document": raw_results['documents'][0][i],
                    "metadata": raw_results['metadatas'][0][i],
                    "score": relevance_score
                })
                
        # Optional: Apply reranking here if needed. 
        # For now, rely on embedding distance.
                
        return formatted_results

# Singleton instance
_searcher = None

def get_hybrid_searcher():
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher
