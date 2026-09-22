import json
from models.llm import LLMModel
from core.conversation_manager import ConversationManager
from core.evidence_manager import EvidenceManager

class ReasoningEngine:
    def __init__(self, llm_model: LLMModel):
        self.llm = llm_model
        
    def analyze_intent(self, user_query: str, chat_history: str, available_image_ids: list) -> dict:
        """
        Analyze what the user is asking, if it's ambiguous, and which images they refer to.
        """
        prompt = f"""
        User Query: "{user_query}"
        
        Recent Conversation History:
        {chat_history}
        
        Available Image IDs (in memory): {available_image_ids}
        
        Determine the following:
        1. "intent": What is the user trying to do? (e.g., "describe", "compare", "follow_up_question")
        2. "referenced_images": Which Image IDs are they referring to? (Infer from words like "this", "previous", "first", or if they just uploaded an image, it's the most recent one). If no images are relevant, return an empty list.
        3. "is_ambiguous": Boolean. Is the query too vague to answer even with the images? (e.g. "Is this good?" without context).
        4. "clarification_needed": If ambiguous, what should we ask the user? (Leave empty if not ambiguous).
        """
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "intent": {"type": "STRING"},
                "referenced_images": {"type": "ARRAY", "items": {"type": "STRING"}},
                "is_ambiguous": {"type": "BOOLEAN"},
                "clarification_needed": {"type": "STRING"}
            },
            "required": ["intent", "referenced_images", "is_ambiguous", "clarification_needed"]
        }
        
        system_prompt = "You are a logical intent analyzer for a multimodal AI system."
        
        return self.llm.generate_structured(prompt, response_schema=schema, system_instruction=system_prompt)

    def generate_response(self, user_query: str, chat_history: str, evidence_text: str) -> str:
        """
        Generate the actual response based on extracted evidence and history.
        """
        prompt = f"""
        Answer the user's query based ONLY on the provided evidence and conversation history.
        
        EVIDENCE:
        {evidence_text}
        
        CONVERSATION HISTORY:
        {chat_history}
        
        USER QUERY: {user_query}
        
        INSTRUCTIONS:
        1. If the evidence does not contain the answer, say that you cannot determine it from the image.
        2. Distinguish between what is directly observed and what is an assumption.
        3. Do NOT hallucinate details not present in the evidence.
        4. Keep the response concise and helpful.
        """
        
        system_prompt = "You are an evidence-based multimodal AI assistant. You answer questions strictly based on the extracted visual evidence provided to you."
        return self.llm.generate_response(prompt, system_instruction=system_prompt)
