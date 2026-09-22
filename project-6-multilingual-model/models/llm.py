import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

class LLMProvider:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        
    def get_llm(self):
        if self.provider.lower() == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in the .env file.")
                
            return ChatGroq(
                groq_api_key=api_key,
                model_name=settings.GROQ_MODEL,
                temperature=0.7,
                max_tokens=1024
            )
        elif self.provider.lower() == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in the .env file.")
                
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=settings.GEMINI_MODEL,
                temperature=0.7,
                max_output_tokens=1024
            )
        else:
            raise NotImplementedError(f"Provider {self.provider} is not fully implemented yet in this demo.")
