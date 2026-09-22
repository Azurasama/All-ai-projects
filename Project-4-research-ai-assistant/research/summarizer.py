import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.llm import get_llm
from research.paper_analyzer import PaperAnalyzer

class Summarizer:
    def __init__(self):
        self.llm = get_llm()
        
    def summarize(self, paper: dict, level: str = "Standard Summary") -> str:
        """
        Generates a summary of the paper at the requested level.
        Levels: 'Quick Summary', 'Standard Summary', 'Detailed Summary', 'Beginner Explanation'
        """
        paper_text = PaperAnalyzer.format_paper_for_llm(paper)
        
        system_prompt = (
            "You are an expert Computer Science researcher. "
            "You will be provided with the metadata and abstract of an arXiv paper. "
            "Your task is to summarize it strictly based on the provided text. "
            "DO NOT invent experimental results, methods, or findings that are not mentioned in the abstract. "
        )
        
        if level == "Quick Summary":
            instructions = "Provide a 2-4 sentence overview of the paper."
        elif level == "Detailed Summary":
            instructions = (
                "Provide a detailed technical explanation. Structure your response with the following headings:\n"
                "- Research Problem\n"
                "- Approach/Methodology\n"
                "- Key Findings\n"
                "- Limitations (only if explicitly stated)\n"
                "- Key Concepts (bulleted list of important technical terms)"
            )
        elif level == "Beginner Explanation":
            instructions = (
                "Explain this paper in beginner-friendly language. Avoid dense jargon where possible, "
                "or explain it simply if it must be used. Use analogies if helpful."
            )
        else: # Standard Summary
            instructions = (
                "Provide a standard summary. Structure your response with the following headings:\n"
                "- Problem\n"
                "- Method\n"
                "- Results\n"
                "- Significance"
            )
            
        prompt = f"Paper Details:\n{paper_text}\n\nInstructions:\n{instructions}"
        
        response = self.llm.generate(prompt, system_prompt=system_prompt)
        return response
