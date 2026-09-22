import sys
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.language_detector import LanguageDetector
from models.llm import LLMProvider
from chatbot.context_manager import ContextManager

class ConversationManager:
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.llm_provider = LLMProvider()
        self.llm = self.llm_provider.get_llm()
        self.context_manager = ContextManager()
        
    def _construct_system_prompt(self, detected_language: str, semantic_context: str) -> str:
        base_instruction = f"""You are a highly intelligent, context-aware Multilingual AI Chatbot.
Your primary capabilities are understanding mixed-language inputs (code-mixing), cross-lingual reasoning, and preserving conversation context even when the language changes.

RULES:
1. ALWAYS respond in the primary language of the user's LATEST message, unless they explicitly ask you to speak in another language.
2. The user's latest message was detected as: {detected_language}.
3. Understand references (like 'it', 'this', 'इसे', 'हा') by looking at the provided recent history and semantic context. The context might be in a different language than the current question, but the underlying meaning remains the same.
4. If a query is ambiguous and cannot be resolved using the provided context, kindly ask for clarification in the detected language.
5. Do not just translate; maintain conversational continuity and reason over the facts discussed previously.

{semantic_context}
"""
        return base_instruction

    def process_message(self, session_id: str, user_message: str) -> dict:
        """
        Main pipeline for processing a turn of conversation.
        """
        # 1. Language Detection
        detected_language = self.language_detector.detect_language(user_message)
        mixed_languages = self.language_detector.detect_mixed_languages(user_message)
        
        # (Optional: simple heuristic for Intent, a real intent model could go here)
        intent = "General Inquiry"
        
        # 2. Get Semantic Context
        semantic_context = self.context_manager.get_semantic_context(session_id, user_message)
        
        # 3. Assemble Prompt with Recent History
        recent_history = self.context_manager.get_recent_history(session_id, k=5)
        
        messages = [SystemMessage(content=self._construct_system_prompt(detected_language, semantic_context))]
        
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=user_message))
        
        # 4. Generate Response using LLM
        response = self.llm.invoke(messages)
        assistant_reply = response.content
        
        # 5. Store in Context Manager
        self.context_manager.add_message(session_id, "user", user_message, detected_language, intent)
        self.context_manager.add_message(session_id, "assistant", assistant_reply, detected_language, "Response")
        
        return {
            "reply": assistant_reply,
            "detected_language": detected_language,
            "mixed_languages": mixed_languages,
            "semantic_context_used": semantic_context != ""
        }
        
    def clear_conversation(self, session_id: str):
        self.context_manager.clear_context(session_id)
