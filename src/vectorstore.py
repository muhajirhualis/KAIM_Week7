# # src/vectorstore.py
# """
# Load the pre-built FAISS vector store.
# """

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from .config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME


# def load_vector_store():
#     """Load pre-built FAISS vector store from disk."""
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     vectorstore = FAISS.load_local(
#         str(VECTOR_STORE_DIR),
#         embeddings,
#         allow_dangerous_deserialization=True  # Required for FAISS
#     )
#     print(f"✓ Vector store loaded from: {VECTOR_STORE_DIR}")
#     return vectorstore


# def get_retriever(vectorstore, k: int = 5):
#     """Create a retriever for finding relevant documents."""
#     retriever = vectorstore.as_retriever(search_kwargs={"k": k})
#     print(f"✓ Retriever created (k={k})")
#     return retriever



# src/vectorstore.py

import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME

CHROMA_PATH = VECTOR_STORE_DIR / "chroma_db"

def load_vector_store():
    """
    Load a pre-persisted ChromaDB vector store from disk.
    Assumes the store was created with the same embedding model.
    """
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_PATH}. "
            "Ensure you've built the index or placed the pre-built store there."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
    print(f"✅ Loaded ChromaDB from: {CHROMA_PATH}")
    return vectorstore

