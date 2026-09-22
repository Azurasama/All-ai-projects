import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class PaperAnalyzer:
    """Utility class to analyze and format paper metadata."""
    
    @staticmethod
    def format_paper_for_llm(paper: dict) -> str:
        """Formats a retrieved paper dictionary into a text string for the LLM prompt."""
        meta = paper.get('metadata', {})
        return f"""
ID: {paper.get('id', 'N/A')}
Title: {meta.get('title', 'N/A')}
Authors: {meta.get('authors', 'N/A')}
Categories: {meta.get('categories', 'N/A')}
Date: {meta.get('update_date', 'N/A')}
Abstract: {paper.get('document', 'N/A')}
"""

    @staticmethod
    def parse_authors(author_string: str) -> list[str]:
        """Basic parsing of author strings."""
        if not author_string:
            return []
        # Usually authors might be comma separated or structured
        return [a.strip() for a in author_string.split(',')]
