class ResponseValidator:
    """Validates the LLM response against hallucination or missing citations."""
    
    @staticmethod
    def validate(response: str, retrieved_papers: list[dict]) -> tuple[bool, str]:
        """
        Simple validation to check if the response mentions the titles or IDs of the retrieved papers.
        Returns (is_valid, reason/modified_response).
        """
        if not retrieved_papers:
            # If no papers were retrieved, the response should ideally say it doesn't know.
            # But we can pass it through.
            return True, response
            
        # Check if any paper title or ID is mentioned
        mentioned_citation = False
        for paper in retrieved_papers:
            title = paper.get('metadata', {}).get('title', '')
            paper_id = paper.get('id', '')
            
            if (title and title.lower() in response.lower()) or (paper_id and paper_id in response):
                mentioned_citation = True
                break
                
        # If there are papers, and the LLM didn't cite them, it might be hallucinating
        # or answering from internal knowledge. 
        # For strict RAG, we might append a warning.
        if not mentioned_citation:
            warning = "\n\n*(Note: The assistant's response did not explicitly cite the retrieved papers. Please verify against sources.)*"
            return False, response + warning
            
        return True, response
