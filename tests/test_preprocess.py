"""
Unit tests for src/preprocess.py
"""
import pytest
import pandas as pd

from src import preprocess, config


class TestCleanComplaintText:
    """Test clean_complaint_text function."""
    
    def test_redaction_removal(self):
        """Test removal of CFPB redaction markers."""
        text = "I was charged XXXX dollars on XX/XX/XXXX and XXXXXX."
        result = preprocess.clean_complaint_text(text)
        
        assert "xxxx" not in result.lower()
        assert "xx/xx" not in result.lower()
        assert "charged" in result
        assert "dollar" in result
    
    def test_boilerplate_removal(self):
        """Test removal of common boilerplate phrases."""
        text = "To whom it may concern, I am writing to file a complaint. Sincerely, John."
        result = preprocess.clean_complaint_text(text)
        
        assert "to whom it may concern" not in result
        assert "i am writing to file a complaint" not in result
        assert "sincerely" not in result
    
    def test_currency_conversion(self):
        """Test currency symbol conversion."""
        text = "I lost $500 and 5% interest."
        result = preprocess.clean_complaint_text(text)
        
        assert "dollar" in result
        assert "percent" in result
        assert "$" not in result
        assert "%" not in result
    
    def test_lowercasing(self):
        """Test text is converted to lowercase."""
        text = "THIS IS ALL CAPS"
        result = preprocess.clean_complaint_text(text)
        
        assert result.islower()
    
    def test_whitespace_normalization(self):
        """Test multiple spaces are normalized."""
        text = "Too    many     spaces"
        result = preprocess.clean_complaint_text(text)
        
        assert "  " not in result
    
    def test_empty_input(self):
        """Test handling of empty/None input."""
        assert preprocess.clean_complaint_text("") == ""
        assert preprocess.clean_complaint_text(None) == ""
        assert preprocess.clean_complaint_text(123) == ""


class TestNormalizeText:
    """Test normalize_text function."""
    
    def test_strip_whitespace(self):
        """Test leading/trailing whitespace is removed."""
        text = "  some text  "
        result = preprocess.normalize_text(text)
        
        assert result == "some text"
    
    def test_multiple_spaces(self):
        """Test multiple spaces are reduced to single space."""
        text = "word1    word2     word3"
        result = preprocess.normalize_text(text)
        
        assert "  " not in result
    
    def test_multiple_newlines(self):
        """Test excessive newlines are reduced."""
        text = "line1\n\n\n\nline2"
        result = preprocess.normalize_text(text)
        
        assert "\n\n\n" not in result


class TestFilterProducts:
    """Test filter_products function."""
    
    def test_filter_keeps_allowed_products(self):
        """Test that allowed products are kept."""
        df = pd.DataFrame({
            "Product": ["Credit card", "Mortgage", "Personal loan"],
            "Data": [1, 2, 3]
        })
        
        allowed = ["Credit card", "Personal loan"]
        result = preprocess.filter_products(df, allowed)
        
        assert len(result) == 2
        assert "Mortgage" not in result["Product"].values
    
    def test_filter_uses_config_defaults(self):
        """Test that function uses config.REQUIRED_PRODUCTS by default."""
        df = pd.DataFrame({
            "Product": ["Credit card", "Mortgage", "Student loan"],
            "Data": [1, 2, 3]
        })
        
        result = preprocess.filter_products(df)
        
        # Should only keep products in REQUIRED_PRODUCTS
        for product in result["Product"].unique():
            assert product in config.REQUIRED_PRODUCTS


class TestStandardizeProducts:
    """Test standardize_products function."""
    
    def test_product_mapping(self):
        """Test that raw labels are mapped to standard categories."""
        df = pd.DataFrame({
            "Product": ["Credit card or prepaid card", "Consumer Loan", "Checking or savings account"]
        })
        
        result = preprocess.standardize_products(df)
        
        assert "Credit card" in result["Product"].values
        assert "Personal loan" in result["Product"].values
        assert "Savings account" in result["Product"].values
    
    def test_unmapped_products_preserved(self):
        """Test that products not in map are preserved."""
        df = pd.DataFrame({
            "Product": ["Unknown Product Type"]
        })
        
        result = preprocess.standardize_products(df)
        
        assert result["Product"].iloc[0] == "Unknown Product Type"


class TestDropEmptyNarratives:
    """Test drop_empty_narratives function."""
    
    def test_drops_empty_strings(self, sample_dataframe_with_empty):
        """Test that empty string narratives are dropped."""
        result = preprocess.drop_empty_narratives(sample_dataframe_with_empty)
        
        # Should only have 2 valid rows
        assert len(result) == 2
    
    def test_drops_none_values(self, sample_dataframe_with_empty):
        """Test that None narratives are dropped."""
        result = preprocess.drop_empty_narratives(sample_dataframe_with_empty)
        
        assert result["Consumer complaint narrative"].isna().sum() == 0
    
    def test_preserves_valid_narratives(self, sample_dataframe_with_empty):
        """Test that valid narratives are preserved."""
        result = preprocess.drop_empty_narratives(sample_dataframe_with_empty)
        
        assert "Valid complaint text" in result["Consumer complaint narrative"].values
        assert "Another valid complaint" in result["Consumer complaint narrative"].values


class TestDropPIIColumns:
    """Test drop_pii_columns function."""
    
    def test_drops_pii_columns(self, sample_dataframe):
        """Test that PII columns are removed."""
        result = preprocess.drop_pii_columns(sample_dataframe)
        
        assert "ZIP code" not in result.columns
        assert "Tags" not in result.columns
    
    def test_preserves_other_columns(self, sample_dataframe):
        """Test that non-PII columns are preserved."""
        result = preprocess.drop_pii_columns(sample_dataframe)
        
        assert "Complaint ID" in result.columns
        assert "Product" in result.columns
        assert "Company" in result.columns


class TestPreprocessData:
    """Test preprocess_data full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        df = pd.DataFrame({
            "Product": ["Credit card", "Consumer Loan", "Mortgage", "Checking or savings account"],
            "Consumer complaint narrative": [
                "To whom it may concern, I have a credit card issue.",
                "Personal loan problem here.",
                "Mortgage issue.",
                ""  # Empty
            ],
            "ZIP code": ["12345", "67890", "11111", "22222"],
            "Tags": [None, None, None, None],
            "Complaint ID": [1, 2, 3, 4],
            "Company": ["A", "B", "C", "D"]
        })
        
        result = preprocess.preprocess_data(df)
        
        # Check filtering worked
        assert "Mortgage" not in result["Product"].values
        
        # Check empty narratives dropped
        assert len(result) == 2  # Only 2 valid rows
        
        # Check PII dropped
        assert "ZIP code" not in result.columns
        
        # Check cleaning applied
        assert "clean_narrative" in result.columns
        assert "narrative_word_count" in result.columns


class TestCreateStratifiedSample:
    """Test create_stratified_sample function."""
    
    def test_stratified_sampling(self):
        """Test that sampling maintains proportions."""
        df = pd.DataFrame({
            "Product": ["Credit card"] * 10 + ["Personal loan"] * 10 + ["Savings account"] * 10,
            "clean_narrative": ["text"] * 30
        })
        
        result = preprocess.create_stratified_sample(df, target_size=15, random_state=42)
        
        assert len(result) == 15
        
        # Check proportions (should be 5 each)
        counts = result["Product"].value_counts()
        assert counts["Credit card"] == 5
        assert counts["Personal loan"] == 5
        assert counts["Savings account"] == 5
    
    def test_returns_full_dataset_if_smaller(self):
        """Test that function returns full dataset if already smaller than target."""
        df = pd.DataFrame({
            "Product": ["Credit card"] * 5,
            "clean_narrative": ["text"] * 5
        })
        
        result = preprocess.create_stratified_sample(df, target_size=10)
        
        assert len(result) == 5
