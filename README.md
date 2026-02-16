# RAG Complaint Chatbot

> Retrieval-Augmented Generation (RAG) system for the CFPB Consumer Complaint Database.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Build Status](https://github.com/zerubabelalpha/rag-complaint-chatbot/actions/workflows/unittest.yml/badge.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)

## Overview

This project implements a robust RAG pipeline designed to analyze and interact with financial consumer complaints. It focuses on five key product categories: **Credit Cards**, **Personal Loans**, **BNPL**, **Savings Accounts**, and **Money Transfers**. 

By leveraging advanced NLP techniques (FAISS + Phi-3 Mini), the system provides semantic search capabilities and intelligent responses to user queries based on real-world complaint data.

## Key Features

- **Interactive Streamlit Dashboard**: Professional UI with real-time streaming responses and business insights side-panel.
- **End-to-End RAG Pipeline**: From raw CSV to vector search and retrieval using LangChain.
- **Source Verification**: Explicit display of source text chunks for every AI response to enhance trust.
- **Advanced Preprocessing**: Automated filtering, PII removal, and text normalization.
- **Comprehensive EDA**: Interactive notebooks for deep data insights and distribution analysis.
- **Vector Search**: FAISS-based vector store for high-performance similarity search.
- **CI/CD Integrated**: Automated verification workflows using GitHub Actions.

---

## Project Architecture

```text
rag-complaint-chatbot/
├── .github/             # CI/CD Workflows
├── dashboard/           # Streamlit Application
│   └── app.py
├── data/                # Data Storage
│   ├── raw/             # Raw CFPB exports
│   ├── processed/       # Cleaned & normalized datasets
│   └── complainet_embeddings.parquet/       # pre built embedding
├── models/              # Trained models & assets
├── notebooks/           # Interactive Development
│   ├── eda.ipynb        # Exploratory Data Analysis
│   └── chunk_embed...   # Pipeline prototyping
├── src/                 # Source Code
│   ├── config.py        # Global configuration
│   ├── preprocess.py    # Cleaning pipelines
│   ├── vectorstore.py   # FAISS management
│   ├── rag_pipeline.py   # Core RAG logic
│   └── ...              # Helper modules
├── tests/               # Verification Suite
└── requirements.txt     # Dependencies
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Git**

### 2. Installation

Clone the repository and set up your environment:

```bash
# Clone the repo
git clone <https://github.com/zerubabelalpha/rag-complaint-chatbot>
cd rag-complaint-chatbot

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
# Note: For Windows CPU-only support
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Data & Index Setup

1.  **Data**: Save your CFPB complaints in `data/raw/complaints.csv`.
2.  **Clean & Filter**: Run `notebooks/eda.ipynb`.
3.  **Build Index**: Run `notebooks/chunk_embed_index.ipynb` to generate embeddings and build the FAISS index.

### 4. Launch the Dashboard

Interact with the chatbot via the Streamlit interface:

```bash
streamlit run dashboard/app.py
```
Then visit the URL displayed in your terminal (usually `http://localhost:8501`).

---

## Modules & Components

| Module | Description |
| :--- | :--- |
| `dashboard/app.py` | Professional Streamlit dashboard with streaming chat and insights. |
| `src/rag_pipeline.py` | Orchestrates retrieval, augmentation, and generation (supports streaming). |
| `src/preprocess.py` | Handles regex-based cleaning, date parsing, and text normalization. |
| `src/vectorstore.py` | Manages FAISS index creation, saving, loading, and semantic search. |
| `src/chunking.py` | Logical text splitting and chunking strategies. |
| `src/config.py` | Centralized configuration for paths, model names, and constants. |

---

## Enhancing Trust: Source Display

To ensure transparency, the system displays the exact snippets used to generate its answer. This allows users to:
1.  **Verify** the AI's claims against original complaint text.
2.  **Reference** specific Complaint IDs and products.
3.  **Understand** the context provided to the LLM.

---

## Verification

Run the optimized core test suite to verify code integrity in seconds:

```bash
# Run the consolidated core tests (recommended)
pytest tests/test_core.py -v

# Run with full coverage report
pytest tests/test_core.py -v --cov=src --cov-report=term-missing
```
