import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipline import RAGPipeline
from src import config

# Page config
st.set_page_config(
    page_title="CFPB Complaint Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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
    # Sidebar for Business Insights
    with st.sidebar:
        st.title("Business Intelligence")
        st.markdown("---")
        
        # Placeholder for dynamic insights
        if "last_docs" in st.session_state and st.session_state.last_docs:
            st.subheader("Current Context Analysis")
            
            # Extract metadata from retrieved docs
            docs = st.session_state.last_docs
            products = [d.metadata.get("product", "Unknown") for d in docs]
            companies = [d.metadata.get("company", "Unknown") for d in docs]
            
            # 1. Product Focus
            df_prod = pd.DataFrame(products, columns=["Product"])
            top_prod = df_prod["Product"].mode()[0] if not df_prod.empty else "N/A"
            
            st.markdown(f"""
            <div class="metric-card">
                <small>Dominant Product Category</small>
                <h3>{top_prod}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Key Companies
            st.markdown("#### Relevant Companies")
            st.dataframe(pd.Series(companies).value_counts().reset_index().rename(columns={"index": "Company", 0: "Count"}), hide_index=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Analyst Note:</b><br>
            These metrics reflect the complaints most relevant to your current query.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("Start chatting to generate business insights based on retrieved context.")
        
        st.markdown("---")
        st.caption("Powered by Phi-3 Mini & RAG")

    # Main Chat Area
    st.markdown('<p class="main-header">Consumer Complaint AI Analyst</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about consumer complaints, trends, and financial products.</p>', unsafe_allow_html=True)

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
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {doc.metadata.get('product', 'N/A')} - {doc.metadata.get('company', 'N/A')}**")
                        st.caption(f"ID: {doc.metadata.get('complaint_id', 'N/A')}")
                        st.text(doc.page_content[:300] + "...")

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
                    stream_generator = pipeline.run_stream(prompt, k=config.RETRIEVAL_CONFIG.k)
                    
                    for item in stream_generator:
                        # Check if it's the source documents chunk
                        if isinstance(item, dict) and "source_documents" in item:
                            source_docs = item["source_documents"]
                            # Update session state for sidebar insights immediately
                            st.session_state.last_docs = source_docs
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
                    
                    # Force rerun to update sidebar with new insights if needed
                    # (Streamlit sometimes lags on sidebar updates from inside callbacks)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Pipeline not initialized.")

if __name__ == "__main__":
    main()
