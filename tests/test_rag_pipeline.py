# tests/test_rag_pipeline.py
import os
import pytest
from src.rag_pipeline import RAGPipeline

@pytest.fixture
def rag():
    # Use dummy token for test (won’t call API unless real token set)
    return RAGPipeline(hf_token=os.getenv("HF_TOKEN", "dummy"))

def test_retrieve_from_parquet(rag):
    # This will fail if parquet doesn’t exist — intentional for CI
    try:
        chunks = rag.retrieve_from_parquet("test", top_k=1)
        assert len(chunks) == 1
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]
    except FileNotFoundError:
        pytest.skip("complaint_embeddings.parquet not found — skip test")

def test_build_prompt(rag):
    chunks = [{"text": "Sample text.", "metadata": {"complaint_id": "123", "product_category": "Credit card"}}]
    prompt = rag.build_prompt("Why?", chunks)
    assert "Context:" in prompt
    assert "Product: Credit card" in prompt
    assert "Question: Why?" in prompt