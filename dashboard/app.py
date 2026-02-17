import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src import config

# Page config
st.set_page_config(
    page_title="CFPB Complaint Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    with st.spinner("Initializing AI Engine and Vector Store..."):
        pipeline = RAGPipeline()
        success = pipeline.initialize()
        if not success:
            st.error("Failed to initialize pipeline. Please check logs.")
            return None
        return pipeline

def main():
    # Main Chat Area
    st.markdown('<p class="main-header">Consumer Complaint AI Analyst</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about consumer complaints, trends, and financial products.</p>', unsafe_allow_html=True)

    # Clean History Button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear Chat", help="Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chatbot
    pipeline = get_pipeline()

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If there are source docs associated with this message (assistant only), show them in expander
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("🔍 View Context Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}: {doc.metadata.get('product', 'N/A')}**")
                        st.markdown(f"> {doc.page_content.strip()}")
                        st.caption(f"Company: {doc.metadata.get('company', 'N/A')} | ID: {doc.metadata.get('complaint_id', 'N/A')}")
                        st.markdown("---")

    # User Input
    if prompt := st.chat_input("Ex: 'What are the common complaints about student loans?'"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            if pipeline:
                message_placeholder = st.empty()
                full_response = ""
                source_docs = []
                
                # Stream response
                # We use the generator from run_stream
                try:
                    # Create a placeholder for sources that appears immediately
                    sources_placeholder = st.empty()
                    
                    stream_generator = pipeline.run_stream(prompt, k=config.RETRIEVAL_CONFIG.k)
                    
                    for item in stream_generator:
                        # Check if it's the source documents chunk
                        if isinstance(item, dict) and "source_documents" in item:
                            source_docs = item["source_documents"]
                            # Display sources immediately
                            with sources_placeholder.container():
                                with st.expander(f"📚 Retrieved {len(source_docs)} relevant documents", expanded=False):
                                    for i, doc in enumerate(source_docs, 1):
                                        st.markdown(f"**{i}. {doc.metadata.get('product', 'N/A')}** ({doc.metadata.get('company', 'N/A')})")
                                        st.caption(doc.page_content[:200] + "...")
                        
                        # Otherwise it's a string token
                        elif isinstance(item, str):
                            full_response += item
                            message_placeholder.markdown(full_response + "▌")
                            
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": source_docs
                    })
                    
                    # Rerun to clear the temporary sources_placeholder and use the history-based one
                    st.rerun()
                    
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Pipeline not initialized.")

if __name__ == "__main__":
    main()
