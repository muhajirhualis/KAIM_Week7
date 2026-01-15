import gradio as gr
import time
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline once
print("Initializing RAG Pipeline (this may take a minute)...")
rag = RAGPipeline()

def predict(message, history):
    """
    Function to handle the chat logic.
    Gradio passes the 'message' (user input) and 'history' (past chat).
    """
    # 1. Run the RAG pipeline
    response = rag.ask(message)
    
    # 2. Format the sources into a readable string
    source_list = []
    for i, doc in enumerate(response.sources):
        # Extract complaint ID and a snippet
        cid = doc.metadata.get("complaint_id", "Unknown")
        snippet = doc.page_content[:200]
        source_list.append(f"**Source {i+1} (ID: {cid}):**\n{snippet}...")
    
    sources_formatted = "\n\n---\n\n".join(source_list)
    
    # 3. Handle Streaming effect (Optional but Recommended)
    # We yield chunks of the answer to make it look like it's typing
    full_answer = response.answer
    for i in range(len(full_answer)):
        time.sleep(0.01)  # Adjust speed here
        yield full_answer[:i+1] + "\n\n**Sources:**\n" + sources_formatted

# Define the Gradio Interface
demo = gr.ChatInterface(
    fn=predict,
    title="🏦 CrediTrust Complaint Assistant",
    description="Ask questions about customer complaints. The system will retrieve relevant data and cite its sources.",
    # REMOVED: theme="soft" (This was causing the error)
    examples=[
        "What are common issues with credit cards?",
        "Why are customers unhappy with personal loans?",
        "Do customers mention hidden fees?"
    ],
    # Note: If retry_btn/undo_btn/clear_btn also throw errors, 
    # you can remove them; Gradio now includes them by default.
)

if __name__ == "__main__":
    # MOVE THE THEME HERE
    demo.launch(theme="soft")