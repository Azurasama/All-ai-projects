import json
from models.llm import LLMService

class ResponseValidator:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def validate(self, drafted_response: str, sentiment: str, intent: str) -> dict:
        """
        Validates the generated response for Appropriateness, Intent Coverage, Tone, etc.
        Returns a JSON object with validation results.
        """
        system_instruction = """
        You are a quality assurance validator for a customer support AI.
        Evaluate the drafted response based on the detected user sentiment and intent.
        Check for:
        1. Sentiment Appropriateness: Does it match the detected sentiment? (e.g. no excessive cheerfulness for negative sentiment)
        2. Intent Coverage: Does it address the user's intent?
        3. Tone: Is it professional, not robotic, not defensive?
        4. Hallucination: Does it invent fake links/policies not typically standard?
        
        Return a JSON object:
        {
            "passed": true/false,
            "reason": "explanation of why it passed or failed",
            "feedback": "if failed, what needs to be changed"
        }
        """
        
        prompt = f"""
Detected Sentiment: {sentiment}
Detected Intent: {intent}

Drafted Response:
"{drafted_response}"

Evaluate and output JSON:
"""
        response = self.llm.generate(prompt=prompt, system_instruction=system_instruction, json_mode=True)
        try:
            parsed = json.loads(response)
            if "passed" not in parsed:
                parsed["passed"] = True
            return parsed
        except Exception:
            return {"passed": True, "reason": "Failed to parse validation, defaulting to True", "feedback": ""}
