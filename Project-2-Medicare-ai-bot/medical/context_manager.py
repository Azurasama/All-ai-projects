class ContextManager:
    def __init__(self):
        self.current_topic = None
        self.recent_entities = {}
        self.previous_question_type = None
        self.history = []

    def update_context(self, user_query: str, extracted_entities: dict, question_type: str):
        # Update topic if we found significant entities (e.g., diseases, organs, ORG/PERSON in basic models)
        significant_labels = ['DISEASE', 'CHEMICAL', 'ORG', 'PERSON', 'GPE']
        new_topic_found = False
        
        for label, entities in extracted_entities.items():
            if label in significant_labels or True: # fallback for generic models
                if entities:
                    self.current_topic = entities[0]
                    self.recent_entities.update(extracted_entities)
                    new_topic_found = True
                    break
        
        if not new_topic_found and not extracted_entities and self.current_topic:
            # If no new entities found, user might be asking a follow up like "what are the symptoms?"
            pass
            
        self.previous_question_type = question_type
        self.history.append({"role": "user", "content": user_query})

    def resolve_query(self, user_query: str) -> str:
        """
        Resolve pronoun/ambiguous queries using context.
        E.g., "What are its symptoms?" -> "What are the symptoms of [current_topic]?"
        """
        query_lower = user_query.lower()
        ambiguous_terms = ["it", "its", "this", "that", "the disease", "the condition"]
        
        has_ambiguity = any(f" {term} " in f" {query_lower} " for term in ambiguous_terms)
        
        # Or if the query is very short and lacks a noun, e.g. "What are the symptoms?"
        words = query_lower.split()
        
        if (has_ambiguity or len(words) < 5) and self.current_topic:
            if " of " not in query_lower and " for " not in query_lower:
                return f"{user_query} of {self.current_topic}"
            else:
                return user_query.replace(" it", f" {self.current_topic}").replace(" its", f" {self.current_topic}'s")
                
        return user_query

    def add_bot_response(self, response: str):
        self.history.append({"role": "assistant", "content": response})
