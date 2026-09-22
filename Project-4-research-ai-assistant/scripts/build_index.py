import csv
import sys
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    PROCESSED_CSV_FILE,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL
)

def build_vector_index():
    if not PROCESSED_CSV_FILE.exists():
        print(f"Error: {PROCESSED_CSV_FILE} not found. Run prepare_data.py first.")
        sys.exit(1)
        
    print("Initializing ChromaDB and Embedding Model...")
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    # Create or get collection
    collection_name = "arxiv_cs_papers"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Papers will be added/updated.")
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection '{collection_name}'.")

    # Load embedding model locally to generate embeddings before passing to Chroma
    # Alternatively, you can use Chroma's built-in embedding functions, but managing
    # it explicitly gives more control.
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Reading processed CSV: {PROCESSED_CSV_FILE}")
    
    # Read CSV and batch process
    BATCH_SIZE = 100
    ids_batch = []
    documents_batch = []
    metadatas_batch = []
    
    # Count rows for tqdm
    with open(PROCESSED_CSV_FILE, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1 # subtract header
        
    with open(PROCESSED_CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(tqdm(reader, total=total_rows, desc="Indexing Papers")):
            paper_id = row['id']
            title = row['title']
            abstract = row['abstract']
            categories = row['categories']
            
            # The text to embed
            # We embed title + abstract for rich semantic meaning
            text_to_embed = f"Title: {title}\nAbstract: {abstract}"
            
            # Metadata for filtering
            metadata = {
                "title": title,
                "authors": row['authors'],
                "categories": categories,
                "update_date": row['update_date']
            }
            
            ids_batch.append(paper_id)
            documents_batch.append(text_to_embed)
            metadatas_batch.append(metadata)
            
            # Process in batches
            if len(ids_batch) >= BATCH_SIZE or (i == total_rows - 1):
                # Generate embeddings
                embeddings_batch = model.encode(documents_batch, show_progress_bar=False).tolist()
                
                # Upsert to ChromaDB
                collection.upsert(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                
                # Clear batches
                ids_batch = []
                documents_batch = []
                metadatas_batch = []
                
    print("Indexing complete! Vector database is ready.")

if __name__ == "__main__":
    build_vector_index()
