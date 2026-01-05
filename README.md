# RAG Complaint Chatbot

A Retrieval-Augmented Generation (RAG) system for analyzing and interacting with the CFPB Consumer Complaint Database. This project focuses on five key financial products: Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.

## Features

- **Data Preprocessing**: Robust pipeline for filtering, PII removal, and text cleaning of complaint narratives.
- **EDA Module**: Comprehensive Exploratory Data Analysis to understand complaint distributions and narrative statistics.
- **RAG Ready**: Cleaned narratives are processed and ready for embedding and vector search.
- **Automated Testing**: CI/CD integration with GitHub Actions for consistent code quality.

## Project Structure

```text
rag-complaint-chatbot/
├── .github/workflows/     # CI/CD pipelines
├── .vscode/               # VS Code project settings
├── data/
│   ├── raw/               # Original CFPB complaints.csv
│   └── processed/         # Filtered and cleaned data
├── notebooks/
│   └── eda.ipynb          # Interactive Exploratory Data Analysis
├── src/
│   ├── config.py          # Project configuration and constants
│   ├── preprocess.py      # Data cleaning and filtering logic
│   ├── eda.py             # EDA helper functions
│   └── ...                # RAG and Vector Store modules
├── tests/                 # Unit and integration tests
└── requirements.txt       # Project dependencies
```

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-complaint-chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   Place the CFPB `complaints.csv` in `data/raw/`.

## Usage

### Exploratory Data Analysis
Open `notebooks/eda.ipynb` in your Jupyter environment to perform initial data analysis and generate the processed dataset.

### Preprocessing
The preprocessing logic is contained in `src/preprocess.py` and is automatically utilized by the EDA notebook and RAG pipeline.

## Testing
Run the verification tests:
```bash
python tests/verify_preprocess.py
```
