import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.llm import get_llm
from research.paper_analyzer import PaperAnalyzer

class PaperComparer:
    def __init__(self):
        self.llm = get_llm()
        
    def compare_papers(self, papers: list[dict]) -> str:
        """
        Generates a comparison between multiple papers based on their metadata and abstract.
        """
        if not papers or len(papers) < 2:
            return "Please provide at least two papers to compare."
            
        papers_text = ""
        for i, paper in enumerate(papers):
            papers_text += f"--- Paper {i+1} ---\n"
            papers_text += PaperAnalyzer.format_paper_for_llm(paper)
            papers_text += "\n"
            
        system_prompt = (
            "You are an expert Computer Science researcher analyzing multiple papers. "
            "Compare the provided papers based ONLY on the information given in their abstracts and metadata. "
            "Do not declare one paper scientifically better unless the evidence explicitly supports it. "
            "Structure your comparison clearly using headings."
        )
        
        prompt = (
            f"Here are the papers to compare:\n\n{papers_text}\n\n"
            "Please compare them across the following dimensions (if information is available):\n"
            "- Research Problem\n"
            "- Methodology/Approach\n"
            "- Datasets/Evaluation\n"
            "- Key Findings/Conclusions"
        )
        
        response = self.llm.generate(prompt, system_prompt=system_prompt)
        return response
