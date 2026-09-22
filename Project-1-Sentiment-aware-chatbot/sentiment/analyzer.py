from models.sentiment_model import SentimentModel
from models.emotion_model import EmotionModel

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_model = SentimentModel()
        self.emotion_model = EmotionModel()

    def analyze(self, text: str) -> dict:
        """
        Runs both sentiment and emotion analysis on the text.
        Returns a dictionary with both results.
        """
        sentiment_res = self.sentiment_model.analyze(text)
        emotion_res = self.emotion_model.analyze(text)
        
        return {
            "sentiment": sentiment_res["label"],
            "sentiment_confidence": round(sentiment_res["score"], 2),
            "emotion": emotion_res["label"],
            "emotion_confidence": round(emotion_res["score"], 2)
        }
