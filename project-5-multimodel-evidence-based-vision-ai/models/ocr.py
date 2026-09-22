import easyocr
import numpy as np
from PIL import Image
import sys

# Fix Windows console encoding issues for EasyOCR's progress bar (blocks char \u2588)
if sys.stdout is not None and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class OCRModel:
    def __init__(self, langs=None):
        if langs is None:
            langs = ['en']
        # Initialize EasyOCR reader
        # Using CPU by default for broader compatibility, but it will use GPU if available
        self.reader = easyocr.Reader(langs, gpu=False)

    def extract_text(self, image: Image.Image) -> list:
        """
        Extract text from a PIL Image using EasyOCR.
        Returns a list of extracted text strings.
        """
        # Convert PIL Image to numpy array (OpenCV format)
        img_np = np.array(image)
        
        # Extract text (returns list of tuples: (bbox, text, prob))
        results = self.reader.readtext(img_np)
        
        # Filter and extract just the text with reasonable confidence
        extracted_text = []
        for bbox, text, prob in results:
            if prob > 0.3:  # Filter out very low confidence extractions
                extracted_text.append(text)
                
        return extracted_text
