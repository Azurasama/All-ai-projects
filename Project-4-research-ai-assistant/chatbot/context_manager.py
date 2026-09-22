class ContextManager:
    """Manages the conversation history and active state."""
    
    def __init__(self):
        self.conversation_history = []
        self.active_papers = [] # List of paper metadata dictionaries
        self.current_topic = None
        self.mentioned_concepts = set()
        
    def add_user_message(self, message: str):
        self.conversation_history.append({"role": "user", "content": message})
        
    def add_assistant_message(self, message: str):
        self.conversation_history.append({"role": "assistant", "content": message})
        
    def set_active_papers(self, papers: list[dict]):
        self.active_papers = papers
        
    def get_active_papers(self) -> list[dict]:
        return self.active_papers
        
    def set_topic(self, topic: str):
        self.current_topic = topic
        
    def add_concept(self, concept: str):
        self.mentioned_concepts.add(concept)
        
    def get_context_string(self) -> str:
        """Returns a string representation of the recent context for the LLM."""
        context = []
        if self.current_topic:
            context.append(f"Current Topic: {self.current_topic}")
            
        if self.active_papers:
            titles = [p.get('metadata', {}).get('title', 'Unknown') for p in self.active_papers[:3]]
            context.append(f"Active Papers: {', '.join(titles)}")
            
        # Add last 3 conversation turns
        history = self.conversation_history[-6:]
        if history:
            context.append("Recent Conversation:")
            for msg in history:
                context.append(f"{msg['role'].capitalize()}: {msg['content']}")
                
        return "\n".join(context)
        
    def clear(self):
        self.conversation_history = []
        self.active_papers = []
        self.current_topic = None
        self.mentioned_concepts = set()
