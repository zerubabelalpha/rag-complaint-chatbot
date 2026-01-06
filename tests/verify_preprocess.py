import sys
from pathlib import Path
import pandas as pd

# allow imports from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess import preprocess_data, create_stratified_sample
from src import config

def test_preprocessing():
    print("Testing Preprocessing Pipeline...")
    # 1. Create mock data based on actual CFPB schema
    mock_data = {
        "Product": [
            "Credit card", 
            "Personal loan", 
            "Mortgage", # Should be filtered out
            "Savings account",
            "Credit card"
        ],
        "Consumer complaint narrative": [
            "I am writing to file a complaint about my credit card. It's too expensive!",
            "I have a personal loan and I want to complain.",
            "Mortgage issue.",
            "", # Should be filtered out (empty)
            None # Should be filtered out (None)
        ],
        "ZIP code": ["32092", "342XX", "12345", "54321", "99999"],
        "Tags": [None, "Older American", None, None, "Servicemember"],
        "Complaint ID": [14195687, 14195688, 14195689, 14195690, 14195691],
        "Company": ["Exp", "East", "Bank", "Savi", "Card"]
    }
    df_raw = pd.DataFrame(mock_data)
    
    # 2. Run preprocessing
    df_processed = preprocess_data(df_raw)
    
    # Check filtering
    remaining_products = df_processed["Product"].unique()
    assert all(p in config.REQUIRED_PRODUCTS for p in remaining_products), "Filter failed"
    assert "Mortgage" not in remaining_products, "Filter failed"
    
    # Check empty narrative dropping
    assert len(df_processed) == 2, f"Expected 2 rows, got {len(df_processed)}"
    
    # Check PII dropping
    assert "ZIP code" not in df_processed.columns, "PII column not dropped"
    
    # Check cleaning
    narrative = df_processed["clean_narrative"].iloc[0]
    assert "i am writing to file a complaint" not in narrative, "Boilerplate not removed"
    assert narrative.islower(), "Text not lowercased"
    
    print("✓ Preprocessing verification PASSED!")

def test_sampling():
    print("\nTesting Stratified Sampling...")
    # Create mock dataset: 30 rows total
    # CC: 10, PL: 10, SA: 10 (Proportion 1/3 each)
    data = {
        "Product": ["Credit card"] * 10 + ["Personal loan"] * 10 + ["Savings account"] * 10,
        "clean_narrative": ["text"] * 30
    }
    df = pd.DataFrame(data)
    
    # Stratify to 15 rows (should get 5 each)
    df_sampled = create_stratified_sample(df, target_size=15)
    
    assert len(df_sampled) == 15, f"Expected 15 samples, got {len(df_sampled)}"
    
    counts = df_sampled["Product"].value_counts()
    print("Sample counts:", counts.to_dict())
    for product in ["Credit card", "Personal loan", "Savings account"]:
        assert counts[product] == 5, f"Incorrect count for {product}: {counts[product]}"
    
    print("✓ Sampling verification PASSED!")

if __name__ == "__main__":
    test_preprocessing()
    test_sampling()
