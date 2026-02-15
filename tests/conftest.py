"""
Pytest fixtures for RAG complaint chatbot tests.
"""
import pytest
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
import tempfile
import shutil


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "Complaint ID": [1, 2, 3],
        "Product": ["Credit card", "Personal loan", "Savings account"],
        "Sub-product": ["General-purpose credit card", "Installment loan", "Checking account"],
        "Issue": ["Late fee", "High interest", "Account closure"],
        "Sub-issue": ["Unexpected fee", "APR too high", "Closed without notice"],
        "Company": ["Bank A", "Bank B", "Bank C"],
        "State": ["CA", "NY", "TX"],
        "Consumer complaint narrative": [
            "I was charged a late fee incorrectly.",
            "The interest rate is too high.",
            "My account was closed without warning."
        ],
        "Date received": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Timely response?": ["Yes", "Yes", "No"],
        "Consumer disputed?": ["No", "Yes", "No"],
        "ZIP code": ["12345", "67890", "11111"],
        "Tags": [None, "Older American", None]
    })


@pytest.fixture
def sample_dataframe_with_empty():
    """Create a sample DataFrame with empty narratives for testing."""
    return pd.DataFrame({
        "Complaint ID": [1, 2, 3, 4],
        "Product": ["Credit card", "Personal loan", "Savings account", "Credit card"],
        "Consumer complaint narrative": [
            "Valid complaint text",
            "",  # Empty string
            None,  # None value
            "Another valid complaint"
        ],
        "Company": ["Bank A", "Bank B", "Bank C", "Bank D"]
    })


@pytest.fixture
def sample_documents():
    """Create sample LangChain Documents for testing."""
    return [
        Document(
            page_content="This is a complaint about credit card fees.",
            metadata={"complaint_id": "1", "product": "Credit card", "company": "Bank A"}
        ),
        Document(
            page_content="I have an issue with my personal loan interest rate.",
            metadata={"complaint_id": "2", "product": "Personal loan", "company": "Bank B"}
        ),
        Document(
            page_content="My savings account was closed without notice.",
            metadata={"complaint_id": "3", "product": "Savings account", "company": "Bank C"}
        )
    ]


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_dataframe):
    """Create a temporary CSV file with sample data."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_processed_dataframe():
    """Create a mock processed DataFrame (after preprocessing)."""
    return pd.DataFrame({
        "Complaint ID": [1, 2, 3],
        "Product": ["Credit card", "Personal loan", "Savings account"],
        "Sub-product": ["General-purpose credit card", "Installment loan", "Checking account"],
        "Issue": ["Late fee", "High interest", "Account closure"],
        "Company": ["Bank A", "Bank B", "Bank C"],
        "State": ["CA", "NY", "TX"],
        "Consumer complaint narrative": [
            "I was charged a late fee incorrectly.",
            "The interest rate is too high.",
            "My account was closed without warning."
        ],
        "clean_narrative": [
            "charged late fee incorrectly",
            "interest rate too high",
            "account closed without warning"
        ],
        "narrative_word_count": [4, 4, 4],
        "Date received": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Timely response?": ["Yes", "Yes", "No"],
        "Consumer disputed?": ["No", "Yes", "No"]
    })
