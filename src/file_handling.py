
import pandas as pd
from typing import Optional
from pathlib import Path

from . import config

def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    if filepath is None:
        filepath = config.RAW_DATA_PATH
    
    # Convert to Path object if string
    filepath = Path(filepath)
    
    # Check if file exists and give helpful error message
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw DATA file not found at: {filepath}\n"
            f"Please place your Kaggle dataset at: {config.RAW_DATA_PATH}"
        )
    
    # Load the CSV
    # low_memory=False prevents mixed type warnings for large files
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"[OK] Loaded {len(df):,} DATA from {filepath.name}")
    return df



def save_processed_data(df: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    if filepath is None:
        filepath = config.PROCESSED_DATA_PATH
    
    filepath = Path(filepath)
    
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    # index=False prevents adding an extra index column
    df.to_csv(filepath, index=False)
    
    print(f"[OK] Saved {len(df):,} DATA to {filepath.name}")
    return filepath



def load_processed_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    if filepath is None:
        filepath = config.PROCESSED_DATA_PATH
    
    # Normalize to Path object
    filepath = Path(filepath)

    # Check if file exists and give helpful error message
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed DATA file not found at: {filepath}\n"
            f"Please run the preprocessing eda notebook  first."
        )

    # Load the CSV
    df = pd.read_csv(filepath, low_memory=False)

    print(f"[OK] Loaded {len(df):,} processed DATA from {filepath.name}")
    return df
