import json
from models.llm import LLMService

class AspectAnalyzer:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def analyze(self, text: str) -> dict:
        """
        Uses the LLM to extract aspect-level sentiments (e.g. "battery" -> "negative", "camera" -> "positive").
        """
        system_instruction = """
        You are an Aspect-Based Sentiment Analysis model.
        Extract specific aspects the user is talking about, and assign a sentiment (positive, negative, neutral) to each aspect.
        Return a JSON object with this schema:
        {
            "aspects": [
                {
                    "aspect": "string (e.g., 'delivery', 'product quality')",
                    "sentiment": "positive | negative | neutral"
                }
            ]
        }
        If no specific aspects are mentioned, return {"aspects": []}.
        """
        
        prompt = f"Customer Message: \"{text}\"\n\nJSON output:"
        
        response = self.llm.generate(prompt=prompt, system_instruction=system_instruction, json_mode=True)
        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            return {"aspects": []}
