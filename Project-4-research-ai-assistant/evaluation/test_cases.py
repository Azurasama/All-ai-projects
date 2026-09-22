# Sample static test cases for evaluating retrieval performance

RETRIEVAL_TEST_CASES = [
    {
        "query": "Attention is all you need",
        "expected_concepts": ["transformer", "attention", "machine translation"],
        # In a real evaluation, we'd have exact expected arXiv IDs here
        # "expected_ids": ["1706.03762"] 
    },
    {
        "query": "Retrieval Augmented Generation for knowledge intensive NLP tasks",
        "expected_concepts": ["RAG", "retrieval", "generation", "NLP"],
    },
    {
        "query": "Graph neural networks for node classification",
        "expected_concepts": ["GNN", "graph", "node classification"],
    },
    {
        "query": "Low-rank adaptation of large language models",
        "expected_concepts": ["LoRA", "LLM", "fine-tuning", "adaptation"],
    }
]

GENERATION_TEST_CASES = [
    {
        "query": "Explain how contrastive learning works.",
        "type": "Concept Explanation"
    },
    {
        "query": "What are the limitations of Transformer models?",
        "type": "Concept Explanation"
    },
    {
        "query": "Summarize the paper about Retrieval Augmented Generation.",
        "type": "Paper Summary"
    }
]
