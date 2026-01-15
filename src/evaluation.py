# src/evaluation.py
import pandas as pd
from .rag_pipeline import RAGPipeline

def run_systematic_evaluation(questions: list):
    rag = RAGPipeline()
    eval_data = []
    
    for q in questions:
        print(f"Evaluating: {q}")
        res = rag.ask(q)
        
        # Summarize first two sources for the table
        source_summary = [
            f"ID: {doc.metadata.get('complaint_id', 'N/A')}" 
            for doc in res.sources[:2]
        ]
        
        eval_data.append({
            "Question": q,
            "Generated Answer": res.answer,
            "Retrieved Sources": ", ".join(source_summary),
            "Quality Score (1-5)": "", # To be filled manually
            "Comments": ""
        })
        
    return pd.DataFrame(eval_data)