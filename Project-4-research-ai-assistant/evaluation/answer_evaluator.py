import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from chatbot.rag_pipeline import RAGPipeline
from chatbot.context_manager import ContextManager
from chatbot.response_validator import ResponseValidator
from evaluation.test_cases import GENERATION_TEST_CASES

def evaluate_generation():
    """
    Runs the LLM generation pipeline on test cases and checks the response validator.
    """
    rag = RAGPipeline()
    
    print("Running Generation Evaluation...")
    print("-" * 40)
    
    for tc in GENERATION_TEST_CASES:
        query = tc["query"]
        print(f"Query: '{query}'")
        
        ctx = ContextManager()
        answer, papers = rag.answer_query(query, ctx)
        
        is_valid, final_answer = ResponseValidator.validate(answer, papers)
        
        print(f"Retrieved {len(papers)} papers.")
        print(f"Validation Passed (Groundedness/Citations): {is_valid}")
        print("Answer Snippet:")
        print(final_answer[:200] + "..." if len(final_answer) > 200 else final_answer)
        print("-" * 40)

if __name__ == "__main__":
    evaluate_generation()
