import os
from config.settings import GEMINI_API_KEY, LLM_MODEL_ID
from google import genai

class LLMService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY is not set. LLM features will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
        self.model_id = LLM_MODEL_ID

    def generate(self, prompt: str, system_instruction: str = None, json_mode: bool = False) -> str:
        if not self.client:
            return '{"error": "API Key not set"}' if json_mode else "API Key not set."

        try:
            config_params = {
                "temperature": 0.2 if json_mode else 0.7,
            }
            if system_instruction:
                config_params["system_instruction"] = system_instruction
            if json_mode:
                config_params["response_mime_type"] = "application/json"
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=genai.types.GenerateContentConfig(**config_params)
            )
            return response.text
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            return f"{{\"error\": \"{str(e)}\"}}" if json_mode else "I'm having trouble processing that right now."
