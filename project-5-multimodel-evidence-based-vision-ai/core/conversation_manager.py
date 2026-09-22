from typing import List, Dict, Any

class ConversationManager:
    def __init__(self):
        # List of message dicts: {"role": "user"|"assistant", "content": str, "image_ids": [str]}
        self.history: List[Dict[str, Any]] = []

    def add_user_message(self, text: str, image_ids: List[str] = None):
        self.history.append({
            "role": "user",
            "content": text,
            "image_ids": image_ids or []
        })

    def add_assistant_message(self, text: str):
        self.history.append({
            "role": "assistant",
            "content": text,
            "image_ids": []
        })

    def get_recent_history(self, turns: int = 5) -> str:
        """
        Format the recent conversation history for the LLM.
        """
        if not self.history:
            return "No previous conversation."
            
        recent = self.history[-(turns*2):]  # multiply by 2 because a turn is usually user+assistant
        
        formatted = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            imgs = f" [Attached Images: {', '.join(msg['image_ids'])}]" if msg.get("image_ids") else ""
            formatted.append(f"{role}{imgs}: {msg['content']}")
            
        return "\n\n".join(formatted)
