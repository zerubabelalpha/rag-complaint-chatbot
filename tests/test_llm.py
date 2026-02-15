"""
Unit tests for src/llm.py
"""
import pytest
from langchain_core.documents import Document

from src import llm, config


class TestFormatDocsForContext:
    """Test format_docs_for_context function."""
    
    def test_formats_single_document(self):
        """Test formatting a single document."""
        docs = [
            Document(
                page_content="This is a complaint about fees.",
                metadata={"complaint_id": "123"}
            )
        ]
        
        result = llm.format_docs_for_context(docs)
        
        assert "[Document 1 | Complaint 123]" in result
        assert "This is a complaint about fees." in result
    
    def test_formats_multiple_documents(self):
        """Test formatting multiple documents."""
        docs = [
            Document(page_content="First complaint", metadata={"complaint_id": "1"}),
            Document(page_content="Second complaint", metadata={"complaint_id": "2"}),
            Document(page_content="Third complaint", metadata={"complaint_id": "3"})
        ]
        
        result = llm.format_docs_for_context(docs)
        
        assert "[Document 1 | Complaint 1]" in result
        assert "[Document 2 | Complaint 2]" in result
        assert "[Document 3 | Complaint 3]" in result
        assert "First complaint" in result
        assert "Second complaint" in result
        assert "Third complaint" in result
    
    def test_handles_missing_complaint_id(self):
        """Test handling documents without complaint_id."""
        docs = [
            Document(page_content="Test", metadata={})
        ]
        
        result = llm.format_docs_for_context(docs)
        
        assert "Unknown ID" in result
    
    def test_separates_documents_with_newlines(self):
        """Test that documents are separated by double newlines."""
        docs = [
            Document(page_content="Doc 1", metadata={"complaint_id": "1"}),
            Document(page_content="Doc 2", metadata={"complaint_id": "2"})
        ]
        
        result = llm.format_docs_for_context(docs)
        
        assert "\n\n" in result
    
    def test_strips_whitespace_from_content(self):
        """Test that content whitespace is stripped."""
        docs = [
            Document(page_content="  Text with spaces  ", metadata={"complaint_id": "1"})
        ]
        
        result = llm.format_docs_for_context(docs)
        
        assert "Text with spaces" in result


class TestRAGPromptTemplate:
    """Test RAG_PROMPT_TEMPLATE structure."""
    
    def test_template_has_required_tags(self):
        """Test that template has Phi-3 specific tags."""
        template = llm.RAG_PROMPT_TEMPLATE
        
        assert "
