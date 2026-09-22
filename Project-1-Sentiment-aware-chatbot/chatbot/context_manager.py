class ContextManager:
    def __init__(self):
        # Stores the full conversation as expected by LLMs (role, content)
        self.messages = []
        
        # Stores enriched metadata for each turn
        self.metadata_history = []

    def add_user_message(self, text: str, sentiment: str, intent: str, emotion: str = None, aspects: list = None):
        self.messages.append({"role": "user", "content": text})
        
        self.metadata_history.append({
            "role": "user",
            "text": text,
            "sentiment": sentiment,
            "intent": intent,
            "emotion": emotion,
            "aspects": aspects or []
        })

    def add_assistant_message(self, text: str):
        self.messages.append({"role": "model", "content": text})
        
        self.metadata_history.append({
            "role": "model",
            "text": text,
            "sentiment": "N/A",
            "intent": "N/A",
            "emotion": "N/A",
            "aspects": []
        })

    def get_conversation_history_formatted(self) -> str:
        formatted = ""
        for msg in self.messages:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def get_recent_sentiments(self) -> list:
        # Extract sentiments from user messages
        sentiments = []
        for meta in self.metadata_history:
            if meta["role"] == "user":
                sentiments.append(meta["sentiment"])
        return sentiments

    def get_current_intent(self) -> str:
        # Find the most recent intent
        for meta in reversed(self.metadata_history):
            if meta["role"] == "user" and meta.get("intent") != "unknown":
                return meta["intent"]
        return "unknown"
