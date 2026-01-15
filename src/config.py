# src/config.py
import os
from pathlib import Path

# 1. Define Project Root
# This reliably finds the root folder regardless of where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 3. Vector Store Path
# This is where your chroma_db folder lives
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma_db"

# 4. Model Configurations
# MUST match the embedding model used in Task 2 (or the parquet file)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# MUST match the model we fixed in llm.py (Mistral v0.3)
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# RAG Parameters
RETRIEVAL_K = 5

# 5. Optional: Hugging Face Cache (good for keeping models organized)
MODELS_DIR = PROJECT_ROOT / "models" / "hf"

def setup_hf_cache():
    """
    Optional: Sets a specific cache directory for downloaded models.
    Call this function at the start of your notebook if you want to use it.
    """
    os.environ["HF_HOME"] = str(MODELS_DIR)
    print(f"HuggingFace cache set to: {MODELS_DIR}")