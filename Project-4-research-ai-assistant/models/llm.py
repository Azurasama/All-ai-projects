import sys
from pathlib import Path
from langchain_community.llms import Ollama
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import LLM_PROVIDER, LLM_MODEL, LLM_API_BASE, LLM_API_KEY

class LLMInterface:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model_name = LLM_MODEL
        
        if self.provider == "ollama":
            self.llm = Ollama(model=self.model_name, base_url=LLM_API_BASE)
        else:
            # Flexible init for providers like openai, anthropic, groq, etc. via langchain
            # Requires appropriate packages installed (e.g., langchain-groq)
            import os
            # Set the API key for the chosen provider if not set in environment
            if LLM_API_KEY:
                # Naive assumption, in reality depends on provider
                os.environ[f"{self.provider.upper()}_API_KEY"] = LLM_API_KEY
            self.llm = init_chat_model(self.model_name, model_provider=self.provider)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generates text from the LLM."""
        if self.provider == "ollama":
            # Ollama basic usage
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            return self.llm.invoke(full_prompt)
        else:
            # Langchain chat model usage
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            response = self.llm.invoke(messages)
            return response.content

# Singleton instance
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterface()
    return _llm_instance
