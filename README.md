# RAG Complaint Chatbot

> A production-ready Retrieval-Augmented Generation (RAG) system for the CFPB Consumer Complaint Database.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Overview

This project implements a robust RAG pipeline designed to analyze and interact with financial consumer complaints. It focuses on five key product categories: **Credit Cards**, **Personal Loans**, **BNPL**, **Savings Accounts**, and **Money Transfers**. 

By leveraging advanced NLP techniques, the system provides semantic search capabilities and intelligent responses to user queries based on real-world complaint data.

## Key Features

- **End-to-End RAG Pipeline**: From raw CSV to vector search and retrieval.
- **Advanced Preprocessing**: Automated filtering, PII removal, and text normalization.
- **Comprehensive EDA**: Interactive notebooks for deep data insights and distribution analysis.
- **Vector Search**: FAISS-based vector store for high-performance similarity search.
- **CI/CD Integrated**: Automated verification workflows using GitHub Actions.

---

## Project Architecture

```text
rag-complaint-chatbot/
├── .github/          # CI/CD Workflows
├── data/             # Data Storage
│   ├── raw/             # Raw CFPB exports
│   └── processed/       # Cleaned & normalized datasets
├── models/           # Trained models & assets
├── notebooks/        # Interactive Development
│   ├── eda.ipynb        # Exploratory Data Analysis
│   └── chunk_embed...   # Pipeline prototyping
├── src/              # Source Code
│   ├── config.py        # Global configuration
│   ├── preprocess.py    # Cleaning pipelines
│   ├── vectorstore.py   # FAISS management
│   └── ...              # Helper modules
├── tests/            # Verification Suite
└── requirements.txt     # Dependencies
```

---

## Quick Start

Follow these steps to get the system up and running in minutes.

### 1. Prerequisities

- **Python 3.11+**
- **Git**

### 2. Installation

Clone the repository and set up your environment:

```bash
# Clone the repo
git clone <repository-url>
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

### 3. Data Setup

1.  Download the **Consumer Complaint Database** from the [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/).
2.  Save the file as `complaints.csv` in the `data/raw/` directory.

### 4. Run the Pipeline

Launch the Jupyter notebooks to process data and build the index:

```bash
jupyter notebook
```

- **Step 1**: Run `notebooks/eda.ipynb` to clean and explore the data.
- **Step 2**: Run `notebooks/chunk_embed_index.ipynb` to generate embeddings and build the FAISS index.

---

## Modules & Components

| Module | Description |
| :--- | :--- |
| `src/preprocess.py` | Handles regex-based cleaning, date parsing, and text normalization. |
| `src/vectorstore.py` | Manages FAISS index creation, saving, loading, and semantic search. |
| `src/chunking.py` | Logical text splitting and chunking strategies. |
| `src/eda.py` | Visualization helpers for plotting distributions and insights. |
| `src/config.py` | Centralized configuration for paths, model names, and constants. |

---

## Verification

Run the included test suite to verify your installation and code integrity:

```bash
# Verify Preprocessing Logic
python tests/verify_preprocess.py

# Verify Document handling
python tests/verify_docs.py

# Verify Text Cleaning
python tests/verify_cleaning.py
```



