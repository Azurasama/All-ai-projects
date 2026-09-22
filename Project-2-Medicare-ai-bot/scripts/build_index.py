import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DATA_DIR
from retrieval.vector_store import ChromaVectorStore
from utils.logging import get_logger

logger = get_logger(__name__)

def build_index():
    input_file = PROCESSED_DATA_DIR / "medquad_processed.json"
    if not input_file.exists():
        logger.error(f"Processed data file not found at {input_file}")
        sys.exit(1)

    logger.info("Loading processed JSON data...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} records.")
    
    store = ChromaVectorStore()
    
    # Check if collection already has data
    if store.collection.count() > 0:
        logger.info(f"Collection already has {store.collection.count()} records. Skipping index build.")
        return

    logger.info("Building ChromaDB index (this may take a while depending on hardware)...")
    
    ids = []
    texts = []
    metadatas = []
    
    for i, item in enumerate(data):
        # We index the question primarily, as users ask questions. 
        # But we could also index "Question: X \n Answer: Y"
        # Let's index the question and the focus to match user semantic queries better.
        text_to_embed = f"Question: {item['question']}\nTopic: {item['focus']}\nAnswer Summary: {item['answer'][:200]}"
        
        # Ensure ID is unique
        unique_id = f"{item['id']}_{i}"
        
        ids.append(unique_id)
        texts.append(text_to_embed)
        metadatas.append({
            "focus": item['focus'],
            "question_type": item['question_type'],
            "source": item['source'],
            "question": item['question'],
            "answer": item['answer']
        })

    store.add_records(ids, texts, metadatas)
    logger.info(f"Successfully added {store.collection.count()} records to the index.")

if __name__ == "__main__":
    build_index()
