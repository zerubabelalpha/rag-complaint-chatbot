"""
Unit tests for src/docs.py
"""
import pytest
import pandas as pd
from langchain_core.documents import Document

from src import docs


class TestRowToDocument:
    """Test row_to_document function."""
    
    def test_converts_row_to_document(self):
        """Test basic row to document conversion."""
        row = pd.Series({
            "Complaint ID": "12345",
            "Product": "Credit card",
            "Sub-product": "General-purpose credit card",
            "Issue": "Late fee",
            "Sub-issue": "Unexpected fee",
            "Company": "Test Bank",
            "State": "CA",
            "clean_narrative": "I was charged a late fee incorrectly.",
            "Date received": "2023-01-01",
            "Timely response?": "Yes",
            "Consumer disputed?": "No"
        })
        
        doc = docs.row_to_document(row)
        
        assert isinstance(doc, Document)
        assert doc.page_content == "I was charged a late fee incorrectly."
        assert doc.metadata["complaint_id"] == "12345"
        assert doc.metadata["product"] == "Credit card"
        assert doc.metadata["company"] == "Test Bank"
    
    def test_handles_missing_values(self):
        """Test handling of NaN/None values in metadata."""
        row = pd.Series({
            "Complaint ID": "123",
            "Product": "Credit card",
            "Sub-product": pd.NA,
            "Issue": None,
            "Company": "Bank",
            "clean_narrative": "Test text"
        })
        
        doc = docs.row_to_document(row)
        
        # NaN/None should be converted to None in metadata
        assert doc.metadata["sub_product"] is None
        assert doc.metadata["issue"] is None
    
    def test_converts_numeric_values(self):
        """Test that numeric values are preserved."""
        row = pd.Series({
            "Complaint ID": 12345,  # Numeric ID
            "Product": "Credit card",
            "clean_narrative": "Text",
            "Company": "Bank"
        })
        
        doc = docs.row_to_document(row)
        
        # Numeric complaint_id should be preserved
        assert doc.metadata["complaint_id"] == 12345
    
    def test_empty_narrative_handling(self):
        """Test handling of empty narrative."""
        row = pd.Series({
            "Complaint ID": "123",
            "Product": "Credit card",
            "clean_narrative": "",
            "Company": "Bank"
        })
        
        doc = docs.row_to_document(row)
        
        assert doc.page_content == ""


class TestDataframeToDocuments:
    """Test dataframe_to_documents function."""
    
    def test_converts_dataframe(self, sample_dataframe):
        """Test converting a full DataFrame to documents."""
        # Add clean_narrative column
        sample_dataframe["clean_narrative"] = sample_dataframe["Consumer complaint narrative"]
        
        documents = docs.dataframe_to_documents(sample_dataframe)
        
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_skips_empty_documents(self, sample_dataframe_with_empty):
        """Test that empty documents are skipped."""
        sample_dataframe_with_empty["clean_narrative"] = sample_dataframe_with_empty["Consumer complaint narrative"]
        
        documents = docs.dataframe_to_documents(sample_dataframe_with_empty)
        
        # Should only have 2 documents (2 empty ones skipped)
        assert len(documents) == 2
    
    def test_preserves_order(self):
        """Test that document order matches DataFrame order."""
        df = pd.DataFrame({
            "Complaint ID": ["1", "2", "3"],
            "Product": ["A", "B", "C"],
            "clean_narrative": ["Text 1", "Text 2", "Text 3"],
            "Company": ["X", "Y", "Z"]
        })
        
        documents = docs.dataframe_to_documents(df)
        
        assert documents[0].metadata["complaint_id"] == "1"
        assert documents[1].metadata["complaint_id"] == "2"
        assert documents[2].metadata["complaint_id"] == "3"
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["Complaint ID", "Product", "clean_narrative", "Company"])
        
        documents = docs.dataframe_to_documents(df)
        
        assert len(documents) == 0


class TestPrintDocumentSample:
    """Test print_document_sample function."""
    
    def test_prints_document(self, capsys):
        """Test that document is printed correctly."""
        doc = Document(
            page_content="This is a test complaint about fees.",
            metadata={"complaint_id": "123", "product": "Credit card"}
        )
        
        docs.print_document_sample(doc)
        
        captured = capsys.readouterr()
        assert "DOCUMENT SAMPLE" in captured.out
        assert "This is a test complaint about fees." in captured.out
        assert "complaint_id: 123" in captured.out
    
    def test_truncates_long_content(self, capsys):
        """Test that long content is truncated."""
        long_text = "word " * 100  # Create long text
        doc = Document(page_content=long_text, metadata={"id": "1"})
        
        docs.print_document_sample(doc, max_content_length=50)
        
        captured = capsys.readouterr()
        assert "..." in captured.out
    
    def test_uses_default_max_length(self, capsys):
        """Test that default max_content_length is used when None."""
        long_text = "a" * 300
        doc = Document(page_content=long_text, metadata={"id": "1"})
        
        docs.print_document_sample(doc, max_content_length=None)
        
        captured = capsys.readouterr()
        # Should use config.DISPLAY_CONFIG.max_content_length (200)
        assert "..." in captured.out
    
    def test_displays_all_metadata(self, capsys):
        """Test that all metadata fields are displayed."""
        doc = Document(
            page_content="Text",
            metadata={"field1": "value1", "field2": "value2", "field3": "value3"}
        )
        
        docs.print_document_sample(doc)
        
        captured = capsys.readouterr()
        assert "field1: value1" in captured.out
        assert "field2: value2" in captured.out
        assert "field3: value3" in captured.out
