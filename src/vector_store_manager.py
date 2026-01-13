import torch
import torchvision
import torchaudio

import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
 
 
class VectorStoreManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the Embedding model and the Text Splitter.
        """
        # Load the embedding model (as recommended in challenge doc)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Recursive splitter handles natural breaks like double newlines and spaces
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_stratified_sample(self, df, sample_size=15000):
        """
        Performs stratified sampling to ensure proportional representation.
        """
        print(f"Sampling {sample_size} records proportionally...")
        
        # Determine sampling fraction
        fraction = sample_size / len(df)
        
        # Group by Product and sample
        sampled_df = df.groupby('Product', group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        )
        
        print(f"Sampled distribution:\n{sampled_df['Product'].value_counts()}")
        return sampled_df

    def create_vector_store(self, df, persist_dir="../vector_store/chroma_db"):
        """
        Chunks text, generates embeddings, and saves to ChromaDB.
        """
        print("Converting narratives to LangChain Documents...")
        documents = []
        for _, row in df.iterrows():
            # Metadata allows us to filter by Product later or find the original ID
            metadata = {
                "Product": row['Product'],
                "Complaint ID": int(row['Complaint ID'])
            }
            doc = Document(
                page_content=row['cleaned_narrative'], 
                metadata=metadata
            )
            documents.append(doc)

        print(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print(f"Indexing to {persist_dir} (this may take a few minutes)...")
        # Initialize and persist the vector store
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        print("Vector store successfully built and persisted.")
        return vector_db