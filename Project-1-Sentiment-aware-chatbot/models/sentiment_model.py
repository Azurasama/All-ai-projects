from transformers import pipeline
import torch
import warnings
from config.settings import SENTIMENT_MODEL_ID

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

class SentimentModel:
    def __init__(self):
        # Use CPU by default, or GPU if available
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.classifier = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID, device=device)
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            self.classifier = None

    def analyze(self, text: str) -> dict:
        if not self.classifier:
            return {"label": "neutral", "score": 1.0}
        
        # Twitter-roberta-base-sentiment-latest returns positive, neutral, negative
        result = self.classifier(text)[0]
        label = result['label'].lower()
        score = result['score']
        return {"label": label, "score": score}
