def calculate_recall_at_k(retrieved_ids: list[str], ground_truth_id: str, k: int) -> int:
    """
    Returns 1 if ground_truth_id is in the top k retrieved_ids, else 0.
    """
    return 1 if ground_truth_id in retrieved_ids[:k] else 0

def calculate_mrr(retrieved_ids: list[str], ground_truth_id: str) -> float:
    """
    Calculates Mean Reciprocal Rank.
    """
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0
