import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid_search import get_hybrid_searcher
from evaluation.test_cases import RETRIEVAL_TEST_CASES

def evaluate_retrieval():
    """
    Evaluates the retrieval system based on how well it finds expected concepts
    in the top-K returned abstracts for specific queries.
    """
    searcher = get_hybrid_searcher()
    
    total_queries = len(RETRIEVAL_TEST_CASES)
    recall_at_5_hits = 0
    
    print("Running Retrieval Evaluation...")
    print("-" * 40)
    
    for tc in RETRIEVAL_TEST_CASES:
        query = tc["query"]
        expected_concepts = [c.lower() for c in tc["expected_concepts"]]
        
        print(f"Query: '{query}'")
        
        # Retrieve top 5
        results = searcher.search(query, top_k=5)
        
        # Check if expected concepts appear in the retrieved abstracts
        concepts_found = set()
        for res in results:
            abstract = res['document'].lower()
            for concept in expected_concepts:
                if concept in abstract:
                    concepts_found.add(concept)
                    
        # Calculate recall for this query
        hit_ratio = len(concepts_found) / len(expected_concepts) if expected_concepts else 0
        print(f"Concepts found: {list(concepts_found)} / {len(expected_concepts)}")
        print(f"Hit ratio: {hit_ratio:.2f}")
        
        if hit_ratio > 0.5: # Consider it a "hit" if >50% of expected concepts are found in top 5
            recall_at_5_hits += 1
            
        print("-" * 40)
        
    overall_recall = recall_at_5_hits / total_queries
    print(f"Overall Recall@5 (Concept Match): {overall_recall:.2f}")

if __name__ == "__main__":
    evaluate_retrieval()
