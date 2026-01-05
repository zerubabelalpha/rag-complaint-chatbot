import re
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split

from . import config


def drop_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PII columns defined in config."""
    df_clean = df.copy()
    cols_to_drop = [c for c in config.PII_COLUMNS_TO_DROP if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"✓ Dropped PII columns: {cols_to_drop}")
    return df_clean


def normalize_text(text: str) -> str:
    """Apply basic text normalization."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def clean_complaint_text(text: Optional[str]) -> str:
    """
    Clean a single complaint narrative to improve quality for embeddings:
    - Lowercase
    - Remove boilerplate text
    - Remove special characters
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"i am writing to (file a complaint|file this complaint|complain)",
        r"i am writing regarding",
        r"to whom it may concern",
        r"please investigate",
        r"if you have any questions",
        r"sincerely",
        r"thank you",
        r"regards",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Remove non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-z0-9\s\.,!?'\-()]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_products(df: pd.DataFrame, products: List[str] = None) -> pd.DataFrame:
    """Keep only records for specified products."""
    if products is None:
        products = config.REQUIRED_PRODUCTS
        
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    for p in products:
        mask = mask | df["Product"].astype(str).str.contains(p, case=False, na=False)

    filtered = df[mask].copy()
    print(f"✓ Filtered products: kept {len(filtered):,} rows for products: {products}")
    return filtered


def drop_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with empty Consumer complaint narrative fields."""
    df = df.copy()
    before = len(df)
    narrative_col = "Consumer complaint narrative"
    
    if narrative_col not in df.columns:
        print(f"⚠ Column '{narrative_col}' not found!")
        return df
        
    df[narrative_col] = df[narrative_col].fillna("")
    df = df[df[narrative_col].astype(str).str.strip() != ""].copy()
    after = len(df)
    print(f"✓ Dropped {before - after:,} rows with empty narratives; {after:,} remain")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full preprocessing according to requirements:
    1. Filter products
    2. Drop empty narratives
    3. Drop PII
    4. Clean narratives
    """
    print("Starting preprocessing pipeline...")
    
    # Filter products
    df = filter_products(df)
    
    # Drop empty narratives
    df = drop_empty_narratives(df)
    
    # Drop PII
    df = drop_pii_columns(df)
    
    # Clean narratives
    df["clean_narrative"] = df["Consumer complaint narrative"].apply(clean_complaint_text)
    print("✓ Cleaned narratives")
    
    # Add word count for EDA
    df["narrative_word_count"] = df["clean_narrative"].apply(lambda x: len(x.split()))
    
    return df


def create_stratified_sample(
    df: pd.DataFrame, 
    target_size: int = 12000, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample of the dataset based on the 'Product' column.
    Ensures proportional representation across product categories.
    
    Args:
        df: Processed DataFrame
        target_size: Desired number of samples (e.g., 10000-15000)
        random_state: Seed for reproducibility
        
    Returns:
        Stratified subset of the original DataFrame
    """
    if len(df) <= target_size:
        print(f"Dataset size ({len(df):,}) is already <= target size ({target_size:,}).")
        return df

    # We use Product as the stratification key
    # If some products have very few samples, train_test_split might fail.
    # But filtering for REQUIRED_PRODUCTS should ensure enough samples.
    _, sample_df = train_test_split(
        df,
        test_size=target_size,
        stratify=df["Product"],
        random_state=random_state,
        shuffle=True
    )
    
    print(f"✓ Created stratified sample: {len(sample_df):,} rows")
    
    # Detailed distribution report
    print("\nProportional Representation Check (Product %):")
    orig_dist = df["Product"].value_counts(normalize=True).sort_index()
    sample_dist = sample_df["Product"].value_counts(normalize=True).sort_index()
    
    report_df = pd.DataFrame({
        "Original %": (orig_dist * 100).round(2),
        "Sample %": (sample_dist * 100).round(2)
    })
    print(report_df)
    
    return sample_df
