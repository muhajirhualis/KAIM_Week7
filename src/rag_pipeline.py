
# src/rag_pipeline.py
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA

# Import from our fixed modules
from .config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, RETRIEVAL_K
from .llm import get_llm

load_dotenv()

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List

class RAGPipeline:
    def __init__(self):
        # 1. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 2. Load Vector Store
        if not VECTOR_STORE_DIR.exists():
            raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_DIR}")
            
        self.vectorstore = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=self.embeddings
        )
        
        # 3. Initialize Chat LLM (from fixed llm.py)
        self.llm = get_llm()
        
        # 4. Define Chat Prompt (Critical for Chat Models)
        # We split instructions into 'system' and user query into 'human'
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst assistant for CrediTrust. 
Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}"""),
            ("human", "{question}"),
        ])

    def ask(self, question: str) -> RAGResponse:
        """Run the end-to-end RAG chain."""
        # RetrievalQA works with Chat Models if the prompt is a ChatPromptTemplate
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        # Invoke
        result = qa_chain.invoke({"query": question})
        
        return RAGResponse(
            question=question,
            answer=result["result"],
            sources=result["source_documents"]
        )