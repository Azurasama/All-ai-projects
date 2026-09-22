import json
from models.llm import LLMService

class IntentModel:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def analyze(self, text: str, context: list = None) -> dict:
        """
        Extracts intent separately from the user's emotion/sentiment.
        """
        system_instruction = """
        You are an expert NLP classifier. Extract the primary intent and urgency of the customer message.
        Ignore their sentiment or emotion - focus purely on what they want to achieve (e.g. order_status, refund, technical_support, complaint).
        Return a JSON object with this schema:
        {
            "intent": "string (short underscore separated)",
            "intent_confidence": float (0.0 to 1.0),
            "urgency": "low" | "medium" | "high" | "critical"
        }
        """
        
        # Build context if available
        context_str = ""
        if context:
            context_str = "Conversation History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-3:]])
        
        prompt = f"{context_str}\n\nCustomer Message: \"{text}\"\n\nJSON output:"
        
        response = self.llm.generate(prompt=prompt, system_instruction=system_instruction, json_mode=True)
        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            return {
                "intent": "unknown",
                "intent_confidence": 0.0,
                "urgency": "low"
            }
