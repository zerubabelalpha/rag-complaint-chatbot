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
- **Intelligent Generation**: Utilizes the Google FLAN-T5-small model to generate professional summaries based strictly on retrieved context.
- **Managed Lifecycle**: A complete preprocessing suite that cleans raw data, removes PII, and handles project configuration.
- **Consumer-Ready Dashboard**: A Streamlit-based interface with real-time streaming and source verification.

## Key Results
- **Metric 1**: ~1.37 Million complaint chunks processed and searchable in milliseconds.
- **Metric 2**: 100% advanced Dashboard
- **Metric 3**: Zero manual data entry required for recurring trend analysis.

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

## Demo
*[GIF or link to dashboard will be provided here]*

## Technical Details
- **Data**: CFPB Consumer Complaint Database. We filter for 5 target products: Credit cards, Personal loans, BNPL, Savings accounts, and Money transfers.
- **Preprocessing**: Robust cleaning pipeline using regex for PII removal, date normalization, and text deduplication.
- **Model**: 
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
  - **LLM**: `google/flan-t5-small` (80M parameters) optimized for low-latency CPU inference.
- **Evaluation**: Unit testing suite with 100% coverage of core RAG logic (Preprocessing, Chunking, Formatting, and Pipeline execution).

## Future Improvements
- **Advanced Re-ranking**: Integrate Cross-Encoders to further refine retrieval results.
- **Agentic Capabilities**: Add tools for the chatbot to generate visualizations of complaint trends dynamically.
- **LLM Scaling**: Support for quantization of larger models (Llama 3, Mistral) for more complex reasoning.
- **Using LLM API**: using remote LLM api with high performance.

## Author
**zerubabelalpha**
- [LinkedIn](https://www.linkedin.com/in/zerubabel-f-52b81931b/)
- [Email](mailto:zerubabelfasika770@gmail.com)
