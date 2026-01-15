# scripts/build_faiss_from_parquet.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

# Paths
parquet_path = Path("data/raw/complaint_embeddings.parquet")
faiss_dir = Path("vector_store/faiss")
faiss_dir.mkdir(parents=True, exist_ok=True)

# Load pre-computed embeddings & metadata
print("Loading embeddings from Parquet...")
df = pd.read_parquet(parquet_path)

# Reconstruct documents
docs = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row["chunk_text"],
        metadata={
            "complaint_id": str(row["complaint_id"]),
            "product_category": row["product_category"],
            "product": row["product"],
            "issue": row["issue"],
            "sub_issue": row.get("sub_issue", ""),
            "company": row["company"],
            "state": row["state"],
            "date_received": row["date_received"],
            "chunk_index": int(row["chunk_index"]),
            "total_chunks": int(row["total_chunks"]),
        }
    )
    docs.append(doc)

# Load embeddings as numpy array
embeddings_np = np.array(df["embedding"].tolist()).astype("float32")

# Create FAISS index
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build LangChain FAISS object
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=None,  # We'll replace it
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

# Manually inject embeddings and docs
vectorstore.index = type(vectorstore).construct_index(embedding_model, embeddings_np)
vectorstore.docstore = InMemoryDocstore(dict(enumerate(docs)))
vectorstore.index_to_docstore_id = {i: i for i in range(len(docs))}

# Save to disk
vectorstore.save_local(str(faiss_dir))
print(f"✅ FAISS index saved to {faiss_dir}")