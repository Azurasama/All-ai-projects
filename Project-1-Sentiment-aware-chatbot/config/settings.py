import os
from dotenv import load_dotenv

load_dotenv()

# Sentiment and Emotion Models
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# LLM Generation Model Settings (using Gemini as a proxy for a fast LLM for the demo)
# Or we could use a local Qwen/Llama if hardware allows. 
# We'll use Gemini since the API is available.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_ID = "gemini-3.6-flash"  # Using a fast generation model

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.50

# Trend Calculation
TREND_WINDOW = 3 # Look at last 3 turns
