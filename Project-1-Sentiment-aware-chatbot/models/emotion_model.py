from transformers import pipeline
import torch
import warnings
from config.settings import EMOTION_MODEL_ID

warnings.filterwarnings("ignore")

class EmotionModel:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.classifier = pipeline("text-classification", model=EMOTION_MODEL_ID, device=device)
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.classifier = None

    def analyze(self, text: str) -> dict:
        if not self.classifier:
            return {"label": "neutral", "score": 1.0}
        
        result = self.classifier(text)[0]
        label = result['label'].lower()
        score = result['score']
        return {"label": label, "score": score}
