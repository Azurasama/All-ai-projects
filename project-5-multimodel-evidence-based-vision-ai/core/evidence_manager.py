import uuid
from typing import Dict, Any, List
from datetime import datetime

class EvidenceManager:
    def __init__(self):
        # image_id -> evidence dict
        self.evidence_store: Dict[str, Dict[str, Any]] = {}

    def store_evidence(self, image_id: str, visual_info: dict, ocr_text: list) -> None:
        """
        Store extracted evidence for a specific image.
        """
        self.evidence_store[image_id] = {
            "image_id": image_id,
            "timestamp": datetime.now().isoformat(),
            "visual_summary": visual_info.get("visual_summary", ""),
            "objects": visual_info.get("objects", []),
            "important_regions": visual_info.get("important_regions", []),
            "confidence": visual_info.get("confidence", "unknown"),
            "ocr_text": "\n".join(ocr_text) if ocr_text else "No text detected.",
            "is_unclear": visual_info.get("is_unclear", False)
        }

    def get_evidence(self, image_id: str) -> Dict[str, Any]:
        """
        Retrieve evidence for a specific image.
        """
        return self.evidence_store.get(image_id, {})

    def get_all_evidence(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored evidence, ordered by timestamp.
        """
        return list(self.evidence_store.values())
        
    def format_evidence_for_prompt(self, relevant_image_ids: List[str] = None) -> str:
        """
        Format evidence into a text block for the LLM.
        """
        if not self.evidence_store:
            return "No images have been analyzed yet."
            
        evidence_texts = []
        ids_to_process = relevant_image_ids if relevant_image_ids else list(self.evidence_store.keys())
        
        for idx, img_id in enumerate(ids_to_process):
            ev = self.get_evidence(img_id)
            if not ev:
                continue
                
            text = f"--- EVIDENCE FOR IMAGE {idx+1} (ID: {img_id}) ---\n"
            text += f"Visual Summary: {ev['visual_summary']}\n"
            text += f"Objects Detected: {', '.join(ev['objects'])}\n"
            text += f"OCR Text Extracted: {ev['ocr_text']}\n"
            text += f"Confidence in extraction: {ev['confidence']}\n"
            
            if ev['is_unclear']:
                text += "WARNING: The vision model indicated this image is unclear or difficult to decipher.\n"
                
            evidence_texts.append(text)
            
        return "\n\n".join(evidence_texts)
