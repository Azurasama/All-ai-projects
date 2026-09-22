import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.llm import get_llm

class QueryClassifier:
    def __init__(self):
        self.llm = get_llm()
        
    def classify(self, query: str) -> str:
        """
        Classifies the user's query into one of the known categories:
        - Paper Search
        - Paper Summary
        - Concept Explanation
        - Paper Comparison
        - Follow-up
        - General/Other
        """
        system_prompt = (
            "You are a routing agent for a Computer Science research assistant. "
            "Classify the following query into EXACTLY ONE of these categories:\n"
            "1. Paper Search (user wants to find papers about a topic)\n"
            "2. Paper Summary (user wants to summarize a specific paper)\n"
            "3. Concept Explanation (user asks 'what is', 'how does X work', etc.)\n"
            "4. Paper Comparison (user wants to compare papers)\n"
            "5. Follow-up (query uses pronouns like 'it', 'they', or refers to previous context)\n"
            "6. General/Other\n\n"
            "Output ONLY the category name."
        )
        
        response = self.llm.generate(query, system_prompt=system_prompt).strip()
        
        # basic normalization
        valid_categories = ["Paper Search", "Paper Summary", "Concept Explanation", "Paper Comparison", "Follow-up", "General/Other"]
        for cat in valid_categories:
            if cat.lower() in response.lower():
                return cat
                
        return "General/Other"
