import os
import glob
import json
import xml.etree.ElementTree as ET
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logging import get_logger

logger = get_logger(__name__)

def process_medquad():
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory {RAW_DATA_DIR} does not exist.")
        sys.exit(1)

    logger.info("Scanning for XML files in MedQuAD...")
    xml_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.xml"), recursive=True)
    logger.info(f"Found {len(xml_files)} XML files.")

    processed_data = []
    skipped_no_answer = 0
    
    for file_path in tqdm(xml_files, desc="Processing XMLs"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # The root is typically <Document>
            # It contains <Focus>, <FocusAnnotations>, <QAPairs>
            focus_elem = root.find("Focus")
            focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else ""
            
            # Source is usually derived from the folder structure or document ID
            source_folder = os.path.basename(os.path.dirname(file_path))
            
            qa_pairs = root.find("QAPairs")
            if qa_pairs is None:
                continue
                
            for qa in qa_pairs.findall("QAPair"):
                question_elem = qa.find("Question")
                answer_elem = qa.find("Answer")
                
                if question_elem is None or answer_elem is None or not answer_elem.text or not answer_elem.text.strip():
                    skipped_no_answer += 1
                    continue
                    
                q_text = question_elem.text.strip() if question_elem.text else ""
                a_text = answer_elem.text.strip()
                q_type = question_elem.get("qtype", "")
                
                record_id = question_elem.get("qid", "")
                if not record_id:
                    record_id = f"{os.path.basename(file_path)}_{qa.get('pid', 'unknown')}"
                
                processed_data.append({
                    "id": record_id,
                    "focus": focus,
                    "question_type": q_type,
                    "question": q_text,
                    "answer": a_text,
                    "source": source_folder,
                    "file_name": os.path.basename(file_path)
                })
        except ET.ParseError:
            logger.warning(f"Failed to parse XML file: {file_path}")
            continue
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info(f"Processed {len(processed_data)} valid Q&A pairs. Skipped {skipped_no_answer} pairs without answers.")
    
    output_path = PROCESSED_DATA_DIR / "medquad_processed.json"
    logger.info(f"Saving processed data to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    logger.info("Data preparation complete.")

if __name__ == "__main__":
    process_medquad()
