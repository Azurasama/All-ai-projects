import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DATA_DIR, ARXIV_DATASET_NAME

def download_kaggle_dataset():
    """
    Downloads the arXiv dataset from Kaggle.
    Requires kaggle package installed and credentials configured.
    """
    print(f"Downloading {ARXIV_DATASET_NAME} from Kaggle into {RAW_DATA_DIR}...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            ARXIV_DATASET_NAME,
            path=RAW_DATA_DIR,
            unzip=True
        )
        print("Download and extraction complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Ensure you have set KAGGLE_USERNAME and KAGGLE_KEY environment variables,")
        print("or placed your kaggle.json in ~/.kaggle/")

if __name__ == "__main__":
    download_kaggle_dataset()
