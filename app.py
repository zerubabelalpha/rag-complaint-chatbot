
import sys
import logging
from pathlib import Path
import gradio as gr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path.cwd()
sys.path.append(str(project_root))

try:
    from src.rag_pipline import RAGPipeline
except ImportError as e:
    logger.error(f"Failed to import RAGPipeline: {e}")
    sys.exit(1)

# Initialize RAG Pipeline
logger.info("Initializing RAG System...")
rag = RAGPipeline()
if not rag.initialize():
    logger.error("Failed to initialize RAG pipeline.")

def process_query(query):
    if not query.strip():
        return "", ""
    
    try:
        result = rag.run(query)
        answer = result.get('answer', "No answer generated.")
        
        # Format sources simply
        sources = "### Sources Used:\n"
        source_docs = result.get('source_documents', [])
        if not source_docs:
            sources += "No specific sources found."
        else:
            for i, doc in enumerate(source_docs, 1):
                cid = doc.metadata.get('complaint_id', 'N/A')
                product = doc.metadata.get('product', 'Unknown')
                company = doc.metadata.get('company', 'Unknown')
                snippet = doc.page_content[:250].replace('\n', ' ') + "..."
                sources += f"**{i}. ID: {cid} | {product} | {company}**\n> {snippet}\n\n"
        
        return answer, sources
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error occurred: {str(e)}", ""

def build_interface():
    with gr.Blocks(title="CFPB Chatbot (Minimal)") as demo:
        gr.Markdown("# CFPB Complaint Chatbot")
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Question", 
                placeholder="Ask about consumer complaints...",
                scale=4
            )
            with gr.Column(scale=1):
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")
        
        answer_box = gr.Textbox(label="AI Answer", interactive=False, lines=6)
        sources_box = gr.Markdown(label="Sources")
        
        # Actions
        ask_btn.click(
            fn=process_query,
            inputs=[user_input],
            outputs=[answer_box, sources_box]
        )
        
        user_input.submit(
            fn=process_query,
            inputs=[user_input],
            outputs=[answer_box, sources_box]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[user_input, answer_box, sources_box]
        )

    return demo

if __name__ == "__main__":
    interface = build_interface()
    interface.launch(server_port=7860)
