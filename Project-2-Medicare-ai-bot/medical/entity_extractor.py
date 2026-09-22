import spacy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NER_MODEL_NAME
from utils.logging import get_logger

logger = get_logger(__name__)

class MedicalEntityExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load(NER_MODEL_NAME)
        except OSError:
            logger.warning(f"Model {NER_MODEL_NAME} not found. Attempting to download...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", NER_MODEL_NAME], check=True)
            self.nlp = spacy.load(NER_MODEL_NAME)

    def extract_entities(self, text: str) -> dict:
        """
        Extract entities from text and categorize them.
        Returns a dictionary of entity labels to list of text spans.
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            # Map standard SpaCy labels to Medical concepts for demonstration if using web_sm
            # In a real medical model (e.g. en_core_sci_sm), labels like DISEASE, CHEMICAL would appear.
            label = ent.label_
            if label not in entities:
                entities[label] = set()
            entities[label].add(ent.text)
        
        # Convert sets to lists for JSON serialization downstream if needed
        return {k: list(v) for k, v in entities.items()}
        
    def detect_question_type(self, text: str) -> str:
        """
        Simple heuristic-based question type detection based on MedQuAD categories.
        """
        text_lower = text.lower()
        if "symptom" in text_lower or "signs" in text_lower:
            return "symptoms"
        elif "treat" in text_lower or "cure" in text_lower or "therapy" in text_lower:
            return "treatment"
        elif "cause" in text_lower or "why" in text_lower:
            return "causes"
        elif "diagnos" in text_lower or "test" in text_lower:
            return "diagnosis"
        elif "prevent" in text_lower:
            return "prevention"
        elif "side effect" in text_lower or "complication" in text_lower:
            return "side effects"
        elif "what is" in text_lower or "define" in text_lower:
            return "definition"
        
        return "general"
