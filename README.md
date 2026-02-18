# RAG Complaint Analyst

A robust Retrieval-Augmented Generation (RAG) system for the CFPB Consumer Complaint Database, designed to provide intelligent, evidence-backed insights into financial consumer issues.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Build Status](https://github.com/zerubabelalpha/rag-complaint-chatbot/actions/workflows/unittest.yml/badge.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)

## Business Problem
Financial institutions and regulatory bodies (like the CFPB) receive hundreds of thousands of consumer complaints. Manually analyzing these to identify trends, specific issues, or company-wide problems is extremely time-consuming and prone to human error. There is a critical need for an automated system that can:
- **Rapidly retrieve** relevant historical complaints based on natural language queries.
- **Synthesize information** into professional, concise summaries.
- **Provide transparency** by citing the exact source documents used for every claim.

## Solution Overview
This project implements an end-to-end RAG pipeline that leverages:
- **Vector Search Engine**: Powered by FAISS and `all-MiniLM-L6-v2` embeddings, enabling semantic search across 1.37 million complaint chunks.
- **Dual LLM Support**: Flexible architecture supporting both local models (Google FLAN-T5-small for CPU inference) and remote API providers (OpenAI-compatible endpoints).
- **Managed Lifecycle**: A complete preprocessing suite that cleans raw data, removes PII, and handles project configuration.
- **Interactive Dashboard**: Streamlit-based chat interface with source document verification, provider switching, and conversation history.

## Key Results
- **1.37M+ Indexed Chunks**: Semantic search across the entire CFPB complaint database with sub-second retrieval times.
- **100% Core Test Coverage**: 8 essential tests validating preprocessing, chunking, retrieval, and generation logic with CI/CD automation.
- **Dual LLM Architecture**: Seamless switching between local CPU inference (FLAN-T5) and remote API providers (OpenAI-compatible).
- **Real-Time Chat Interface**: Interactive Streamlit dashboard with source document citations, conversation history, and transparent evidence tracking.

## Quick Start
```bash
# Clone the repo
git clone https://github.com/zerubabelalpha/rag-complaint-chatbot
cd rag-complaint-chatbot

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies (Windows CPU-optimized)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Launch the Dashboard
streamlit run dashboard/app.py
```

## Project Structure

```text
rag-complaint-chatbot/
├── .github/             # CI/CD Workflows
├── dashboard/           # Streamlit Application
│   └── app.py
├── data/                # Data Storage
│   ├── raw/             # Raw CFPB exports
│   ├── processed/       # Cleaned & normalized datasets
│   └── complaint_embeddings.parquet/       # pre built embedding
├── models/              # Trained models & assets
├── notebooks/           # Interactive Development
│   ├── eda.ipynb        # Exploratory Data Analysis
│   ├── chunk_embed_index.ipynb   # Pipeline prototyping
│   └── rag_pipeline.ipynb        # RAG Pipeline Evaluation
├── src/                 # Source Code
│   ├── config.py        # Global configuration
│   ├── preprocess.py    # Cleaning pipelines
│   ├── vectorstore.py   # FAISS management
│   ├── rag_pipeline.py   # Core RAG logic
│   └── ...              # Helper modules
├── tests/               # Verification Suite
└── requirements.txt     # Dependencies
```

## Core Features

### RAG Pipeline
- **Semantic Search**: FAISS-powered vector similarity search with `all-MiniLM-L6-v2` embeddings (384 dimensions)
- **Context Retrieval**: Top-k document retrieval with configurable relevance thresholds
- **Evidence-Based Generation**: LLM responses strictly grounded in retrieved complaint data
- **Source Transparency**: Every answer includes citations with complaint IDs, companies, and product types

### Dual LLM Architecture
- **Local Inference**: Google FLAN-T5-small (80M parameters) optimized for CPU-only environments
- **Remote API**: OpenAI-compatible endpoint support with configurable base URLs and models
- **Dynamic Switching**: Change providers on-the-fly through the dashboard without restarting
- **Error Handling**: Comprehensive error messages with debugging guidance for API failures

### Interactive Dashboard
- **Real-Time Chat**: Streamlit-based conversational interface with message history
- **Source Verification**: Expandable panels showing retrieved documents with metadata
- **Provider Configuration**: Sidebar controls for LLM selection and API credentials
- **Session Management**: Clear chat history and maintain conversation context

### Data Preprocessing
- **PII Removal**: Regex-based detection and redaction of sensitive information (SSN, phone numbers, emails)
- **Text Normalization**: Whitespace cleanup, date standardization, and deduplication
- **Product Filtering**: Focus on 5 key financial products (Credit cards, Personal loans, BNPL, Savings, Money transfers)
- **Chunking Strategy**: Intelligent text splitting with overlap to preserve context

### Testing & CI/CD
- **8 Core Tests**: 100% coverage of critical RAG logic (preprocessing, chunking, retrieval, generation)
- **Fast Mocking**: Dependency injection for embeddings, LLM, and vector store (tests run in seconds)
- **Automated CI**: GitHub Actions workflow with pytest integration
- **Quality Assurance**: Validates data integrity, prompt formatting, and end-to-end pipeline flow

## Technical Details
- **Data**: CFPB Consumer Complaint Database. We filter for 5 target products: Credit cards, Personal loans, BNPL, Savings accounts, and Money transfers.
- **Preprocessing**: Robust cleaning pipeline using regex for PII removal, date normalization, and text deduplication.
- **Models**: 
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
  - **Local LLM**: `google/flan-t5-small` (80M parameters) optimized for low-latency CPU inference.
  - **Remote LLM**: OpenAI-compatible API support with configurable endpoints and models.
- **Evaluation**: Comprehensive unit testing suite with 8 core tests covering 100% of critical RAG logic:
  - Configuration validation
  - Text preprocessing and PII removal
  - Document conversion and chunking
  - File I/O operations
  - LLM prompt formatting
  - Vector store operations (mocked for speed)
  - Full RAG pipeline integration

## Future Improvements
- **Advanced Re-ranking**: Integrate Cross-Encoders to further refine retrieval results and improve answer relevance.
- **Agentic Capabilities**: Add tools for the chatbot to generate visualizations of complaint trends dynamically (charts, graphs, timelines).
- **LLM Scaling**: Support for quantization of larger models (Llama 3, Mistral, Gemma) for more complex reasoning and nuanced responses.
- **Query Analytics**: Track common user queries, response quality metrics, and retrieval performance for continuous improvement.
- **Multi-Language Support**: Extend preprocessing and generation to handle complaints in multiple languages.
- **Hybrid Search**: Combine semantic search with keyword-based BM25 for improved retrieval accuracy.

## Author
**zerubabelalpha**
- [LinkedIn](https://www.linkedin.com/in/zerubabel-f-52b81931b/)
- [Email](mailto:zerubabelfasika770@gmail.com)
