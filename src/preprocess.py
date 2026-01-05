import re
import pandas as pd
from typing import List, Optional

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
