import pytest
from core.conversation_manager import ConversationManager
from core.evidence_manager import EvidenceManager

# In a real environment, you would use unittest.mock to mock the LLM responses
# or run integration tests directly against the Gemini API if the key is present.

def test_evidence_manager_storage():
    """Test 2: Evidence extraction storage"""
    em = EvidenceManager()
    em.store_evidence(
        image_id="img_123",
        visual_info={"visual_summary": "A chart showing sales.", "objects": ["chart", "bars"], "is_unclear": False},
        ocr_text=["Sales 2023", "Q1: 500"]
    )
    
    evidence = em.get_evidence("img_123")
    assert evidence["image_id"] == "img_123"
    assert "Sales 2023" in evidence["ocr_text"]
    assert "chart" in evidence["objects"]

def test_conversation_manager_context():
    """Test 4 & 8: Context Retention & Cross-Turn Reasoning"""
    cm = ConversationManager()
    cm.add_user_message("What is this image?", ["img_123"])
    cm.add_assistant_message("It is a chart.")
    cm.add_user_message("What was the highest value?", [])
    
    history = cm.get_recent_history()
    assert "What is this image?" in history
    assert "img_123" in history
    assert "What was the highest value?" in history

def test_ambiguity_handling_logic(monkeypatch):
    """Test 6: Ambiguous Query (Mocking ReasoningEngine)"""
    from core.reasoning_engine import ReasoningEngine
    from models.llm import LLMModel
    
    # Mock LLM to simulate ambiguity detection
    class MockLLM:
        def generate_structured(self, *args, **kwargs):
            return {
                "intent": "unknown",
                "referenced_images": [],
                "is_ambiguous": True,
                "clarification_needed": "Better in terms of what?"
            }
            
    engine = ReasoningEngine(MockLLM())
    result = engine.analyze_intent("Which one is better?", "", ["img_1", "img_2"])
    
    assert result["is_ambiguous"] is True
    assert "Better in terms of what?" in result["clarification_needed"]

def test_evidence_validation_logic():
    """Test 7: Evidence Validation"""
    from core.response_validator import ResponseValidator
    from models.llm import LLMModel
    
    class MockLLM:
        def generate_structured(self, prompt, **kwargs):
            # If the response hallucinates a name not in evidence
            if "John" in prompt:
                return {
                    "is_supported": False,
                    "has_hallucination": True,
                    "addresses_query": True,
                    "correction_needed": "The name John is not mentioned in the evidence."
                }
            return {"is_supported": True, "has_hallucination": False, "addresses_query": True, "correction_needed": ""}
            
    validator = ResponseValidator(MockLLM())
    
    # Simulated hallucination
    bad_result = validator.validate("The document is signed by John.", "Who signed it?", "The document is signed, but the signature is illegible.")
    assert bad_result["has_hallucination"] is True
    assert bad_result["is_supported"] is False

def test_image_reference_resolution():
    """Test 9: Image Reference Resolution"""
    from core.reasoning_engine import ReasoningEngine
    from models.llm import LLMModel
    
    class MockLLM:
        def generate_structured(self, *args, **kwargs):
            return {
                "intent": "comparison",
                "referenced_images": ["img_1", "img_2"],
                "is_ambiguous": False,
                "clarification_needed": ""
            }
            
    engine = ReasoningEngine(MockLLM())
    result = engine.analyze_intent("Compare the first image to the second one.", "User uploaded img_1 and img_2.", ["img_1", "img_2"])
    
    assert "img_1" in result["referenced_images"]
    assert "img_2" in result["referenced_images"]
