"""
Unit tests for src/chunking.py
"""
import pytest
from langchain_core.documents import Document

from src import chunking, config


class TestCreateTextSplitter:
    """Test create_text_splitter function."""
    
    def test_default_parameters(self):
        """Test splitter creation with default config values."""
        splitter = chunking.create_text_splitter()
        
        assert splitter._chunk_size == config.CHUNK_SIZE
        assert splitter._chunk_overlap == config.CHUNK_OVERLAP
    
    def test_custom_parameters(self):
        """Test splitter creation with custom parameters."""
        splitter = chunking.create_text_splitter(chunk_size=1000, chunk_overlap=100)
        
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 100
    
    def test_separators_configured(self):
        """Test that separators are properly configured."""
        splitter = chunking.create_text_splitter()
        
        assert splitter._separators == ["\n\n", "\n", ". ", " ", ""]


class TestChunkDocuments:
    """Test chunk_documents function."""
    
    def test_chunks_long_documents(self):
        """Test that long documents are split into chunks."""
        long_text = "word " * 200  # Create a long document
        docs = [Document(page_content=long_text, metadata={"id": "1"})]
        
        chunks = chunking.chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        
        # Should create multiple chunks
        assert len(chunks) > 1
    
    def test_preserves_metadata(self):
        """Test that metadata is preserved in chunks."""
        docs = [Document(
            page_content="Some text here",
            metadata={"complaint_id": "123", "product": "Credit card"}
        )]
        
        chunks = chunking.chunk_documents(docs)
        
        assert chunks[0].metadata["complaint_id"] == "123"
        assert chunks[0].metadata["product"] == "Credit card"
    
    def test_adds_chunk_index(self):
        """Test that chunk_index is added to metadata."""
        long_text = "word " * 200
        docs = [Document(page_content=long_text, metadata={"complaint_id": "1"})]
        
        chunks = chunking.chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        
        # Check chunk indices
        for i, chunk in enumerate(chunks):
            if chunk.metadata["complaint_id"] == "1":
                assert "chunk_index" in chunk.metadata
                assert isinstance(chunk.metadata["chunk_index"], int)
    
    def test_multiple_documents(self, sample_documents):
        """Test chunking multiple documents."""
        chunks = chunking.chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        
        # Should have at least as many chunks as documents
        assert len(chunks) >= len(sample_documents)
    
    def test_short_documents_not_split(self):
        """Test that short documents are not split."""
        docs = [Document(page_content="Short text", metadata={"id": "1"})]
        
        chunks = chunking.chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        
        # Should remain as 1 chunk
        assert len(chunks) == 1


class TestGetChunkStats:
    """Test get_chunk_stats function."""
    
    def test_calculates_stats(self):
        """Test that statistics are correctly calculated."""
        chunks = [
            Document(page_content="a" * 100),
            Document(page_content="b" * 200),
            Document(page_content="c" * 150)
        ]
        
        stats = chunking.get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["min_length"] == 100
        assert stats["max_length"] == 200
        assert stats["mean_length"] == 150.0
        assert stats["median_length"] == 150
    
    def test_empty_chunks_list(self):
        """Test handling of empty chunks list."""
        stats = chunking.get_chunk_stats([])
        
        assert "error" in stats
    
    def test_single_chunk(self):
        """Test stats with single chunk."""
        chunks = [Document(page_content="test text")]
        
        stats = chunking.get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == 1
        assert stats["min_length"] == stats["max_length"]


class TestPrintChunkSamples:
    """Test print_chunk_samples function."""
    
    def test_prints_without_error(self, sample_documents, capsys):
        """Test that function prints without errors."""
        chunking.print_chunk_samples(sample_documents, n_samples=2)
        
        captured = capsys.readouterr()
        assert "SAMPLE CHUNKS" in captured.out
        assert "Chunk 1" in captured.out
    
    def test_respects_n_samples_limit(self, sample_documents, capsys):
        """Test that n_samples parameter limits output."""
        chunking.print_chunk_samples(sample_documents, n_samples=1)
        
        captured = capsys.readouterr()
        assert "Chunk 1" in captured.out
        assert "Chunk 2" not in captured.out
    
    def test_handles_more_samples_than_chunks(self, capsys):
        """Test requesting more samples than available chunks."""
        docs = [Document(page_content="Single doc")]
        
        chunking.print_chunk_samples(docs, n_samples=10)
        
        captured = capsys.readouterr()
        assert "Showing 1 of 1" in captured.out
