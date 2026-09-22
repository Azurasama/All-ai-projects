import os
from google import genai
from google.genai import types
from PIL import Image
import json
from config.settings import GEMINI_API_KEY, VISION_MODEL_NAME

class VisionModel:
    def __init__(self):
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = genai.Client()

    def extract_visual_information(self, image: Image.Image) -> dict:
        """
        Extract structured visual information from an image.
        Returns a dictionary representing visual observations.
        """
        prompt = """
        Analyze this image in detail and extract visual information.
        Do NOT guess context that is not visible. 
        Provide the output in JSON format with the following keys:
        - "objects": list of prominent objects detected
        - "visual_summary": a clear, objective summary of the image
        - "important_regions": list of interesting visual areas or features
        - "confidence": string ("high", "medium", "low") indicating your confidence in the visual analysis
        - "is_unclear": boolean, true if the image is extremely blurry or impossible to decipher
        """
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                "visual_summary": {"type": "STRING"},
                "important_regions": {"type": "ARRAY", "items": {"type": "STRING"}},
                "confidence": {"type": "STRING"},
                "is_unclear": {"type": "BOOLEAN"}
            },
            "required": ["objects", "visual_summary", "important_regions", "confidence", "is_unclear"]
        }
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1
        )
        
        try:
            response = self.client.models.generate_content(
                model=VISION_MODEL_NAME,
                contents=[image, prompt],
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Vision model error: {e}")
            return {
                "objects": [],
                "visual_summary": "Error analyzing image.",
                "important_regions": [],
                "confidence": "low",
                "is_unclear": True
            }
