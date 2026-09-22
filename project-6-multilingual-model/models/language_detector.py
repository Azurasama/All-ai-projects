from lingua import Language, LanguageDetectorBuilder

class LanguageDetector:
    def __init__(self):
        # We explicitly support English, Hindi, Marathi, and Spanish
        self.languages = [
            Language.ENGLISH, 
            Language.HINDI, 
            Language.MARATHI, 
            Language.SPANISH
        ]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()

    def detect_language(self, text: str) -> str:
        """
        Detects the primary language of the given text.
        Returns the language name as a string.
        """
        if not text or not text.strip():
            return "Unknown"
            
        language = self.detector.detect_language_of(text)
        if language is None:
            return "Unknown"
        return language.name.capitalize()

    def detect_mixed_languages(self, text: str):
        """
        Returns confidence values for multiple languages if the text is code-mixed.
        """
        confidence_values = self.detector.compute_language_confidence_values(text)
        # Filter out very low confidence matches
        significant_langs = [(lang.language.name.capitalize(), lang.value) for lang in confidence_values if lang.value > 0.1]
        
        if not significant_langs:
            return [{"language": "Unknown", "confidence": 1.0}]
            
        return [{"language": lang, "confidence": conf} for lang, conf in significant_langs]
