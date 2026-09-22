import os
from utils.logging import get_logger

logger = get_logger(__name__)

class LLM:
    def __init__(self):
        # We use a mock LLM as requested, returning canned responses built from evidence.
        self.is_mock = True
        logger.info("Initialized Mock LLM.")

    def generate_answer(self, prompt: str, evidence: list) -> str:
        """
        In a real app, this would call Groq, OpenAI, or a local pipeline.
        Since we are doing a Mock implementation:
        We will construct an answer directly from the evidence summary.
        """
        if not evidence:
            return "I couldn't find sufficiently relevant information in the available MedQuAD knowledge base to answer that reliably."
            
        # Mocking an LLM response by aggregating the answers from the evidence
        logger.info("Generating mock response based on evidence.")
        
        # Check if the prompt implies we have evidence
        response_parts = []
        for i, ev in enumerate(evidence[:3]):
            ans_excerpt = ev['metadata'].get('answer', '')[:400] + "..."
            response_parts.append(f"- Based on MedQuAD (Focus: {ev['metadata'].get('focus')}): {ans_excerpt}")
            
        final_answer = (
            "Based on the retrieved medical evidence, here is the information:\n\n" +
            "\n\n".join(response_parts) +
            "\n\n*Note: This is a synthesized response using a mock LLM. In production, an actual LLM would fluently summarize this text.*"
        )
        
        return final_answer
