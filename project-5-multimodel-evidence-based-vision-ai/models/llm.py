import os
from google import genai
from google.genai import types
from PIL import Image
import json
from config.settings import GEMINI_API_KEY, REASONING_MODEL_NAME

class LLMModel:
    def __init__(self):
        # We assume GEMINI_API_KEY is available in the environment or passed directly
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = genai.Client() # Assumes configured via environment

    def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generate a text response using the reasoning LLM.
        """
        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction
            
        response = self.client.models.generate_content(
            model=REASONING_MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text

    def generate_structured(self, prompt: str, response_schema: dict, system_instruction: str = None) -> dict:
        """
        Generate a structured JSON response based on a schema.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )
        if system_instruction:
            config.system_instruction = system_instruction
            
        response = self.client.models.generate_content(
            model=REASONING_MODEL_NAME,
            contents=prompt,
            config=config
        )
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("Failed to decode JSON from LLM")
            return {}
