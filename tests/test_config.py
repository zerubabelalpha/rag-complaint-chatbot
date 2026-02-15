"""
Unit tests for src/config.py
"""
import pytest
from pathlib import Path
import os

from src import config


class TestDataclasses:
    """Test configuration dataclasses."""
    
    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        cfg = config.ChunkingConfig()
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
    
    def test_chunking_config_custom(self):
        """Test ChunkingConfig with custom values."""
        cfg = config.ChunkingConfig(chunk_size=1000, chunk_overlap=100)
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 100
    
    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig default values."""
        cfg = config.RetrievalConfig()
        assert cfg.k == 3
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        cfg = config.LLMConfig()
        assert cfg.model_name == "microsoft/Phi-3-mini-4k-instruct"
        assert cfg.temperature == 0.1
        assert cfg.max_new_tokens == 512
    
    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        cfg = config.DisplayConfig()
        assert cfg.content_preview_length == 300
        assert cfg.max_content_length == 200
        assert cfg.separator_length == 60


class TestConfigurationInstances:
    """Test module-level configuration instances."""
    
    def test_chunking_config_instance(self):
        """Test CHUNKING_CONFIG instance exists."""
        assert isinstance(config.CHUNKING_CONFIG, config.ChunkingConfig)
        assert config.CHUNK_SIZE == config.CHUNKING_CONFIG.chunk_size
        assert config.CHUNK_OVERLAP == config.CHUNKING_CONFIG.chunk_overlap
    
    def test_retrieval_config_instance(self):
        """Test RETRIEVAL_CONFIG instance exists."""
        assert isinstance(config.RETRIEVAL_CONFIG, config.RetrievalConfig)
        assert config.RETRIEVAL_K == config.RETRIEVAL_CONFIG.k
    
    def test_llm_config_instance(self):
        """Test LLM_CONFIG instance exists."""
        assert isinstance(config.LLM_CONFIG, config.LLMConfig)
    
    def test_display_config_instance(self):
        """Test DISPLAY_CONFIG instance exists."""
        assert isinstance(config.DISPLAY_CONFIG, config.DisplayConfig)


class TestPaths:
    """Test path configurations."""
    
    def test_project_root_exists(self):
        """Test PROJECT_ROOT is a valid path."""
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()
    
    def test_data_directories(self):
        """Test data directory paths are defined."""
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.RAW_DATA_DIR, Path)
        assert isinstance(config.PROCESSED_DATA_DIR, Path)
    
    def test_model_directory(self):
        """Test models directory path is defined."""
        assert isinstance(config.MODELS_DIR, Path)


class TestSetupHFCache:
    """Test HuggingFace cache setup function."""
    
    def test_setup_hf_cache_creates_directory(self, tmp_path, monkeypatch):
        """Test that setup_hf_cache creates the models directory."""
        # Mock MODELS_DIR to use temp directory
        test_models_dir = tmp_path / "test_models"
        monkeypatch.setattr(config, "MODELS_DIR", test_models_dir)
        
        # Run setup
        config.setup_hf_cache()
        
        # Verify directory was created
        assert test_models_dir.exists()
        assert test_models_dir.is_dir()
    
    def test_setup_hf_cache_sets_env_vars(self, tmp_path, monkeypatch):
        """Test that setup_hf_cache sets environment variables."""
        test_models_dir = tmp_path / "test_models"
        monkeypatch.setattr(config, "MODELS_DIR", test_models_dir)
        
        config.setup_hf_cache()
        
        # Check environment variables
        assert os.environ["HF_HOME"] == str(test_models_dir)
        assert os.environ["TRANSFORMERS_CACHE"] == str(test_models_dir)
        assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(test_models_dir)


class TestProductConfiguration:
    """Test product mapping and filtering configuration."""
    
    def test_product_map_structure(self):
        """Test PRODUCT_MAP has correct structure."""
        assert isinstance(config.PRODUCT_MAP, dict)
        assert len(config.PRODUCT_MAP) == 5
        
        # Check all target products are keys
        for product in config.TARGET_PRODUCTS:
            assert product in config.PRODUCT_MAP
    
    def test_required_products_list(self):
        """Test REQUIRED_PRODUCTS is correctly flattened."""
        assert isinstance(config.REQUIRED_PRODUCTS, list)
        assert len(config.REQUIRED_PRODUCTS) > 0
        
        # Verify it contains items from PRODUCT_MAP
        assert "Credit card" in config.REQUIRED_PRODUCTS
        assert "Consumer Loan" in config.REQUIRED_PRODUCTS
