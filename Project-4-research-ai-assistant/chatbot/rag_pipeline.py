import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.llm import get_llm
from retrieval.hybrid_search import get_hybrid_searcher
from chatbot.context_manager import ContextManager
from research.paper_analyzer import PaperAnalyzer

class RAGPipeline:
    def __init__(self):
        self.searcher = get_hybrid_searcher()
        self.llm = get_llm()
        
    def answer_query(self, query: str, context_mgr: ContextManager) -> tuple[str, list]:
        """
        Executes the RAG pipeline.
        Returns the (generated_answer, retrieved_papers).
        """
        # 1. Retrieve relevant papers
        papers = self.searcher.search(query, top_k=5)
        
        # 2. Update context
        if papers:
            context_mgr.set_active_papers(papers)
            
        # 3. Format context for LLM
        evidence_text = ""
        for i, paper in enumerate(papers):
            evidence_text += f"\n--- Evidence {i+1} ---\n"
            evidence_text += PaperAnalyzer.format_paper_for_llm(paper)
            
        conv_context = context_mgr.get_context_string()
        
        system_prompt = (
            "You are an expert Computer Science research assistant. "
            "Answer the user's question using ONLY the provided evidence. "
            "If the evidence does not contain the answer, explicitly state that the available research collection does not contain enough relevant information. "
            "Always cite the 'Evidence ID' or Title when making claims."
        )
        
        prompt = (
            f"Conversation Context:\n{conv_context}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"User Question: {query}\n"
        )
        
        # 4. Generate answer
        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        
        return answer, papers
