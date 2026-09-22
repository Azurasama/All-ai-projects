import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.llm import get_llm
from research.paper_analyzer import PaperAnalyzer

class ConceptExtractor:
    def __init__(self):
        self.llm = get_llm()
        
    def extract_concepts(self, paper: dict) -> dict:
        """
        Extracts key technical terms, methodologies, and datasets from the paper.
        Returns a dictionary.
        """
        paper_text = PaperAnalyzer.format_paper_for_llm(paper)
        
        system_prompt = (
            "You are an expert NLP system specialized in Computer Science terminology. "
            "Extract entities from the provided abstract. "
            "Output strictly valid JSON with the following keys: "
            "'algorithms' (list of strings), 'models' (list of strings), 'datasets' (list of strings), 'metrics' (list of strings). "
            "If an entity type is not found, return an empty list for that key."
        )
        
        prompt = f"Paper:\n{paper_text}\n\nOutput JSON:"
        
        # Try to parse the LLM output as JSON
        try:
            response = self.llm.generate(prompt, system_prompt=system_prompt)
            # Find the JSON part if there is surrounding text
            # This is a naive extraction, assuming the LLM wraps it in ```json ... ``` or just outputs dict
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except Exception as e:
            print(f"Failed to extract concepts: {e}")
            return {"algorithms": [], "models": [], "datasets": [], "metrics": []}
