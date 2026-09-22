from models.llm import LLMService
from models.intent_model import IntentModel
from sentiment.analyzer import SentimentAnalyzer
from sentiment.aspect_analyzer import AspectAnalyzer
from sentiment.trend_analyzer import TrendAnalyzer
from chatbot.context_manager import ContextManager
from chatbot.response_strategy import ResponseStrategy
from chatbot.response_generator import ResponseGenerator
from chatbot.response_validator import ResponseValidator

class ConversationManager:
    def __init__(self):
        self.llm_service = LLMService()
        self.intent_model = IntentModel(self.llm_service)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_analyzer = AspectAnalyzer(self.llm_service)
        
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator(self.llm_service)
        self.response_validator = ResponseValidator(self.llm_service)
        
        self.last_debug_info = {}

    def process_message(self, text: str) -> str:
        # 1. Sentiment & Emotion Analysis
        sentiment_res = self.sentiment_analyzer.analyze(text)
        sentiment = sentiment_res["sentiment"]
        emotion = sentiment_res["emotion"]
        
        # 2. Intent Detection
        intent_res = self.intent_model.analyze(text, self.context_manager.messages)
        intent = intent_res.get("intent", "unknown")
        
        # 3. Aspect Analysis
        aspect_res = self.aspect_analyzer.analyze(text)
        aspects = aspect_res.get("aspects", [])
        
        # 4. Context & Trend
        self.context_manager.add_user_message(text, sentiment, intent, emotion, aspects)
        trend = TrendAnalyzer.calculate_trend(self.context_manager.get_recent_sentiments())
        
        # 5. Response Strategy
        strategy = ResponseStrategy.get_strategy(sentiment, emotion, trend)
        
        # 6. Response Generation
        history_formatted = self.context_manager.get_conversation_history_formatted()
        
        MAX_RETRIES = 2
        final_response = "I'm sorry, I'm having trouble processing that right now."
        validation_passed = False
        validation_reason = ""
        
        for attempt in range(MAX_RETRIES):
            draft = self.response_generator.generate_response(strategy, history_formatted, intent)
            
            # 7. Response Validation
            val_res = self.response_validator.validate(draft, sentiment, intent)
            validation_passed = val_res.get("passed", True)
            validation_reason = val_res.get("reason", "")
            
            if validation_passed:
                final_response = draft
                break
            else:
                # If failed, adjust strategy for the retry
                strategy += f"\n\nNote: The previous draft failed validation. Feedback: {val_res.get('feedback', '')}. Please correct it."
                final_response = draft # fallback to draft if last retry fails
                
        # 8. Save Assistant message
        self.context_manager.add_assistant_message(final_response)
        
        # Store debug info for UI
        self.last_debug_info = {
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_res.get("sentiment_confidence"),
            "emotion": emotion,
            "emotion_confidence": sentiment_res.get("emotion_confidence"),
            "intent": intent,
            "intent_confidence": intent_res.get("intent_confidence"),
            "urgency": intent_res.get("urgency"),
            "aspects": aspects,
            "trend": trend,
            "validation_passed": validation_passed,
            "validation_reason": validation_reason,
            "strategy_used": strategy
        }
        
        return final_response
