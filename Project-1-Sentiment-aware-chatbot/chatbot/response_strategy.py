class ResponseStrategy:
    @staticmethod
    def get_strategy(sentiment: str, emotion: str, trend: str) -> str:
        """
        Determines the system instruction/strategy based on sentiment and emotion.
        """
        base_strategy = "You are a customer support AI for a company. Your goal is to solve the customer's intent effectively."
        
        style = ""
        
        if sentiment == "positive":
            style = "Response Characteristics: Friendly, Appreciative, Helpful, Natural. Acknowledge the positive experience naturally, but do not overdo the enthusiasm. If the user still has an intent to resolve (e.g. a support request), solve it without assuming a positive message means there is no problem."
        elif sentiment == "neutral":
            style = "Response Characteristics: Clear, Direct, Informative, Professional. Focus purely on answering the user's request accurately."
        elif sentiment == "negative":
            style = "Response Characteristics: Empathetic, Calm, Solution-focused. Acknowledge frustration if present. Avoid unnecessary repetition. Do not just say 'I detect negative sentiment.' Help them work through the issue."
            
            if emotion in ["anger", "frustration", "sadness"]:
                style += " The user seems particularly frustrated or upset. Acknowledge the issue, avoid defensive language, prioritize resolution. Do NOT automatically claim a human agent will intervene."
                
        if trend == "Declining":
            style += " Note: The customer's sentiment has been declining over the last few turns. Be extra careful, concise, and focused on resolution."
            
        return f"{base_strategy}\n\n{style}\n\nSeparate the customer's feeling from their intent. Focus on resolving the intent while matching the tone dictated above."
