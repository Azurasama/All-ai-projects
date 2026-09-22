import os
import requests
import zipfile
import io
import sys
import shutil
from pathlib import Path

# Add the project root to sys.path so we can import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DATA_DIR
from utils.logging import get_logger

logger = get_logger(__name__)

ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"

def download_medquad():
    logger.info("Starting MedQuAD dataset download from zip...")
    
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        logger.info(f"Directory {RAW_DATA_DIR} is not empty. Assuming data is already downloaded.")
        return

    RAW_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {ZIP_URL}...")
        response = requests.get(ZIP_URL)
        response.raise_for_status()
        
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(RAW_DATA_DIR.parent)
        
        # Rename the extracted folder 'MedQuAD-master' to 'MedQuAD'
        extracted_folder = RAW_DATA_DIR.parent / "MedQuAD-master"
        if extracted_folder.exists():
            if RAW_DATA_DIR.exists():
                shutil.rmtree(RAW_DATA_DIR)
            extracted_folder.rename(RAW_DATA_DIR)
            
        logger.info(f"Successfully downloaded and extracted MedQuAD dataset to {RAW_DATA_DIR}.")
    except Exception as e:
        logger.error(f"Failed to download or extract repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_medquad()
