"""
Consolidated core functionality tests for RAG Chatbot.
These tests provide near-instant verification of essential business logic.
"""
import pytest
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from src import (
    preprocess, chunking, vectorstore, 
    rag_pipeline, config, docs, file_handling, llm
)

@pytest.mark.core
class TestCoreLogic:
    """Consolidated tests for the core RAG components."""

    # --- Config ---
    def test_config_essentials(self):
        """Verify essential configuration constants."""
        assert config.PROJECT_ROOT.exists()
        assert config.CHUNK_SIZE > 0
        assert config.LLM_CONFIG.model_name is not None

    # --- Preprocessing ---
    def test_preprocess_essentials(self):
        """Verify basic text cleaning logic."""
        text = "I was charged $100 incorrectly [REDACTED]"
        cleaned = preprocess.clean_complaint_text(text)
        assert "[REDACTED]" not in cleaned
        assert "charged" in cleaned
        assert "100" in cleaned  # Ensure numeric data is preserved/handled

    # --- Documents ---
    def test_doc_conversion_essentials(self, sample_dataframe):
        """Verify DataFrame to Document conversion."""
        sample_dataframe["clean_narrative"] = sample_dataframe["Consumer complaint narrative"]
        documents = docs.dataframe_to_documents(sample_dataframe)
        assert len(documents) == 3
        assert isinstance(documents[0], Document)
        assert documents[0].metadata["complaint_id"] == 1

    # --- Chunking ---
    def test_chunking_essentials(self, sample_documents):
        """Verify document chunking logic."""
        chunks = chunking.chunk_documents(sample_documents, chunk_size=50)
        assert len(chunks) >= len(sample_documents)
        assert "chunk_index" in chunks[0].metadata

    # --- File Handling ---
    def test_file_handling_essentials(self, temp_data_dir, sample_dataframe):
        """Verify file load/save capabilities."""
        output_path = temp_data_dir / "test_core_save.csv"
        file_handling.save_processed_data(sample_dataframe, output_path)
        assert output_path.exists()
        
        loaded_df = file_handling.load_processed_data(output_path)
        assert len(loaded_df) == len(sample_dataframe)

    # --- LLM Utilities ---
    def test_llm_formatting_essentials(self, sample_documents):
        """Verify prompt formatting logic."""
        context = llm.format_docs_for_context(sample_documents)
        assert "Content: This is a complaint" in context
        assert sample_documents[0].page_content in context

    # --- Vector Store (Mocked) ---
    def test_vectorstore_essentials(self, sample_documents, mock_embeddings, mock_faiss):
        """Verify vectorstore search interface using mocks."""
        vs = vectorstore.create_vector_store(sample_documents)
        results = vectorstore.search_similar(vs, "query")
        assert len(results) > 0
        assert results[0].page_content == "Mock doc"

    # --- Full RAG Pipeline (Mocked) ---
    def test_rag_pipeline_full_flow(self, mock_llm, mock_faiss):
        """Verify the complete RAG flow from query to answer using mocks."""
        from unittest.mock import patch
        with patch("src.vectorstore.load_vector_store") as mock_load:
            mock_load.return_value = mock_faiss
            
            pipeline = rag_pipeline.RAGPipeline()
            pipeline.is_initialized = True
            pipeline.vector_store = mock_faiss
            pipeline.llm_engine = mock_llm
            
            result = pipeline.run("test query")
            assert result["answer"] == "Mocked LLM Answer"
            assert "source_documents" in result
            assert len(result["source_documents"]) > 0
