import json
import csv
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    RAW_JSON_FILE,
    PROCESSED_CSV_FILE,
    CS_CATEGORIES,
    MAX_PAPERS_TO_PROCESS
)

def process_data():
    """
    Reads the massive JSONL file line by line, filters for CS papers,
    extracts relevant fields, and saves to a CSV.
    """
    if not RAW_JSON_FILE.exists():
        print(f"Error: {RAW_JSON_FILE} not found. Run download_data.py first.")
        sys.exit(1)
        
    print(f"Filtering CS papers from {RAW_JSON_FILE}...")
    print(f"Saving to {PROCESSED_CSV_FILE}")
    
    cs_category_keys = set(CS_CATEGORIES.keys())
    
    processed_count = 0
    total_scanned = 0
    
    # Define fields to extract
    fieldnames = [
        'id', 'title', 'authors', 'categories', 
        'abstract', 'update_date', 'doi'
    ]
    
    with open(RAW_JSON_FILE, 'r', encoding='utf-8') as f_in, \
         open(PROCESSED_CSV_FILE, 'w', encoding='utf-8', newline='') as f_out:
         
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        # Since the file is large, we don't know the exact line count, but we can track progress
        for line in tqdm(f_in, desc="Scanning JSONL"):
            total_scanned += 1
            
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # Paper categories are space-separated
            paper_categories = set(paper.get('categories', '').split(' '))
            
            # Check if there is any intersection with our CS categories
            if paper_categories.intersection(cs_category_keys):
                # Clean up fields
                # Remove newlines from abstract and title
                abstract = paper.get('abstract', '').replace('\n', ' ').strip()
                title = paper.get('title', '').replace('\n', ' ').strip()
                
                row = {
                    'id': paper.get('id', ''),
                    'title': title,
                    'authors': paper.get('authors', ''),
                    'categories': paper.get('categories', ''),
                    'abstract': abstract,
                    'update_date': paper.get('update_date', ''),
                    'doi': paper.get('doi', '')
                }
                
                writer.writerow(row)
                processed_count += 1
                
                if processed_count >= MAX_PAPERS_TO_PROCESS:
                    print(f"Reached MAX_PAPERS_TO_PROCESS limit ({MAX_PAPERS_TO_PROCESS}). Stopping early.")
                    break

    print(f"Processing complete!")
    print(f"Total lines scanned: {total_scanned}")
    print(f"Total CS papers saved: {processed_count}")

if __name__ == "__main__":
    process_data()
