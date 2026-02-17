import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

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
def get_pipeline(provider="local"):
    """Initialize and cache the RAG pipeline."""
    with st.spinner(f"Initializing AI Engine ({provider}) and Vector Store..."):
        pipeline = RAGPipeline()
        success = pipeline.initialize(provider=provider)
        if not success:
            st.error("Failed to initialize pipeline. Please check logs.")
            return None
        return pipeline

def main():
    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ LLM Settings")
        
        provider = st.selectbox(
            "Select LLM Provider",
            options=["local", "openai"],
            index=0 if config.LLM_CONFIG.provider == "local" else 1,
            help="Choose between running a small model locally on your CPU or using an external API."
        )
        
        # Track provider changes to force re-initialization
        if "current_provider" not in st.session_state:
            st.session_state.current_provider = provider
            
        if st.session_state.current_provider != provider:
            st.session_state.current_provider = provider
            # Force clear cache for get_pipeline if provider changes
            get_pipeline.clear()
            st.rerun()

        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", value=config.LLM_CONFIG.openai_api_key, type="password")
            base_url = st.text_input("Base URL", value=config.LLM_CONFIG.openai_base_url)
            model_name = st.text_input("Model Name", value=config.LLM_CONFIG.openai_model)
            
            # Update config instance (eagerly)
            config.LLM_CONFIG.openai_api_key = api_key
            config.LLM_CONFIG.openai_base_url = base_url
            config.LLM_CONFIG.openai_model = model_name
            config.LLM_CONFIG.provider = "openai"
        else:
            config.LLM_CONFIG.provider = "local"

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
    pipeline = get_pipeline(provider=provider)

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history (Same as before)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
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
                try:
                    with st.spinner("Analyzing complaints..."):
                        # Use synchronous run instead of run_stream
                        result = pipeline.run(prompt, k=config.RETRIEVAL_CONFIG.k)
                        full_response = result.get("answer", "")
                        source_docs = result.get("source_documents", [])

                    # Display retrieved sources in an expander
                    if source_docs:
                        with st.expander(f"📚 Retrieved {len(source_docs)} relevant documents", expanded=False):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**{i}. {doc.metadata.get('product', 'N/A')}** ({doc.metadata.get('company', 'N/A')})")
                                st.caption(doc.page_content[:200] + "...")

                    # Display the final answer
                    st.markdown(full_response)
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": source_docs
                    })
                    st.rerun()

                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Pipeline not initialized.")

if __name__ == "__main__":
    main()
