# src/ingest_data.py
import pandas as pd
import pyarrow.parquet as pq
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, RAW_DATA_DIR

def build_vector_store_from_parquet():
    parquet_path = RAW_DATA_DIR / "complaint_embeddings.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    # 1. Initialize Embeddings and Chroma Client
    print(f"--- Initializing ChromaDB at {VECTOR_STORE_DIR} ---")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection_name = "complaints_index"

    # 2. Open Parquet file as a stream (does NOT load into memory yet)
    parquet_file = pq.ParquetFile(parquet_path)
    total_row_groups = parquet_file.num_row_groups
    
    print(f"--- Starting Streaming Ingestion ({total_row_groups} Row Groups) ---")

    # 3. Process Row Group by Row Group
    for i in range(total_row_groups):
        # Read only one row group at a time into RAM
        table_batch = parquet_file.read_row_group(i)
        batch_df = table_batch.to_pandas()
        
        # Prepare data for Chroma
        ids = [str(idx) for idx in batch_df.index]
        texts = batch_df['text'].tolist()
        embeddings = batch_df['embedding'].tolist()
        
        # Clean metadata (Chroma requires dicts with simple types)
        # Filter for only necessary columns to save space
        metadatas = batch_df[['complaint_id', 'product', 'issue', 'company']].to_dict('records')

        # Add this batch to the Vector Store
        vector_db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings_model
        )
        
        vector_db.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # Print progress every row group
        print(f"✅ Processed Row Group {i+1}/{total_row_groups} (~{len(batch_df)} records)")

    print(f"\n🎉 Success! Vector store built successfully without memory crash.")

if __name__ == "__main__":
    build_vector_store_from_parquet()