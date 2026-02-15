"""
Unit tests for src/file_handling.py
"""
import pytest
import pandas as pd
from pathlib import Path

from src import file_handling


class TestLoadRawData:
    """Test load_raw_data function."""
    
    def test_load_raw_data_success(self, sample_csv_file):
        """Test loading a valid CSV file."""
        df = file_handling.load_raw_data(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Complaint ID" in df.columns
    
    def test_load_raw_data_file_not_found(self, temp_data_dir):
        """Test loading a non-existent file raises FileNotFoundError."""
        non_existent_file = temp_data_dir / "does_not_exist.csv"
        
        with pytest.raises(FileNotFoundError):
            file_handling.load_raw_data(non_existent_file)
    
    def test_load_raw_data_with_path_object(self, sample_csv_file):
        """Test that function accepts Path objects."""
        assert isinstance(sample_csv_file, Path)
        df = file_handling.load_raw_data(sample_csv_file)
        assert isinstance(df, pd.DataFrame)


class TestSaveProcessedData:
    """Test save_processed_data function."""
    
    def test_save_processed_data_success(self, temp_data_dir, sample_dataframe):
        """Test saving a DataFrame to CSV."""
        output_path = temp_data_dir / "processed_output.csv"
        
        result_path = file_handling.save_processed_data(sample_dataframe, output_path)
        
        # Verify file was created
        assert output_path.exists()
        assert result_path == output_path
        
        # Verify content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_save_processed_data_creates_parent_dirs(self, temp_data_dir, sample_dataframe):
        """Test that parent directories are created if they don't exist."""
        nested_path = temp_data_dir / "nested" / "dir" / "output.csv"
        
        file_handling.save_processed_data(sample_dataframe, nested_path)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_save_processed_data_no_index(self, temp_data_dir, sample_dataframe):
        """Test that saved CSV doesn't include index column."""
        output_path = temp_data_dir / "no_index.csv"
        
        file_handling.save_processed_data(sample_dataframe, output_path)
        
        loaded_df = pd.read_csv(output_path)
        # If index was saved, there would be an extra 'Unnamed: 0' column
        assert "Unnamed: 0" not in loaded_df.columns


class TestLoadProcessedData:
    """Test load_processed_data function."""
    
    def test_load_processed_data_success(self, temp_data_dir, sample_dataframe):
        """Test loading a processed CSV file."""
        # First save a file
        csv_path = temp_data_dir / "processed.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Then load it
        df = file_handling.load_processed_data(csv_path)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
    
    def test_load_processed_data_file_not_found(self, temp_data_dir):
        """Test loading a non-existent file raises FileNotFoundError."""
        non_existent_file = temp_data_dir / "missing.csv"
        
        with pytest.raises(FileNotFoundError):
            file_handling.load_processed_data(non_existent_file)
    
    def test_load_processed_data_preserves_columns(self, temp_data_dir, sample_dataframe):
        """Test that all columns are preserved during save/load cycle."""
        csv_path = temp_data_dir / "columns_test.csv"
        
        # Save and load
        file_handling.save_processed_data(sample_dataframe, csv_path)
        loaded_df = file_handling.load_processed_data(csv_path)
        
        # Check columns match
        assert list(loaded_df.columns) == list(sample_dataframe.columns)
