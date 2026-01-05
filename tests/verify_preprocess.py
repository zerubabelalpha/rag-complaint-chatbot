import sys
from pathlib import Path
import pandas as pd

# allow imports from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess import preprocess_data
from src import config

def test_preprocessing():
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
    
    print("Raw Data Shape:", df_raw.shape)
    
    # 2. Run preprocessing
    df_processed = preprocess_data(df_raw)
    
    print("\nProcessed Data Shape:", df_processed.shape)
    print("Columns in processed data:", df_processed.columns.tolist())
    
    # Check filtering
    remaining_products = df_processed["Product"].unique()
    print("Remaining Products:", remaining_products)
    assert all(p in config.REQUIRED_PRODUCTS for p in remaining_products), "Filter failed: kept non-required products"
    assert "Mortgage" not in remaining_products, "Filter failed: kept Mortgage"
    
    # Check empty narrative dropping
    # Rows 0, 1, 3, 4 are kept after product filtering.
    # Row 3 (Savings) has empty narrative -> Dropped
    # Row 4 (Credit card) has None narrative -> Dropped
    # So only rows 0 and 1 remain.
    assert len(df_processed) == 2, f"Expected 2 rows, got {len(df_processed)}"
    
    # Check PII dropping
    assert "ZIP code" not in df_processed.columns, "PII column not dropped"
    assert "Tags" not in df_processed.columns, "PII column not dropped"
    
    # Check cleaning
    narrative = df_processed["clean_narrative"].iloc[0]
    print("\nCleaned Narrative Sample:", narrative)
    assert "i am writing to file a complaint" not in narrative, "Boilerplate not removed"
    assert narrative.islower(), "Text not lowercased"
    
    print("\n✓ Preprocessing verification PASSED!")

if __name__ == "__main__":
    test_preprocessing()
