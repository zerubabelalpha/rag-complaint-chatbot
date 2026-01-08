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
    print(f"[OK] Dropped PII columns: {cols_to_drop}")
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
    Clean a single complaint narrative to improve quality for RAG:
    - Removes CFPB redaction markers (XXXX)
    - Removes common boilerplate filler
    - Standardizes currency and numbers
    - Normalizes whitespace and basic punctuation
    """
    if not isinstance(text, str):
        return ""

    # 1. Standardize lowercasing
    text = text.lower()

    # 2. Remove common CFPB redaction markers
    # Matches XXXX, XX/XX, XX/XX/XXXX, XXXXXXXX, etc.
    text = re.sub(r'x{2,}(/\w{2,})*(/\w{2,})*', ' ', text)
    text = re.sub(r'xx+', ' ', text)

    # 3. Expand boilerplate removal
    boilerplate_patterns = [
        r"i am writing to (file a complaint|file this complaint|complain|dispute)",
        r"i am writing regarding",
        r"to whom it may concern",
        r"please investigate",
        r"if you have any questions",
        r"i would like to",
        r"thank you for your time",
        r"sincerely",
        r"thank you",
        r"regards",
        r"at your earliest convenience",
        r"attached please find",
        r"consumer complaint narrative",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # 4. Standardize currency and common noise
    text = text.replace("$", " dollar ")
    text = text.replace("%", " percent ")
    
    # 5. Filter characters: keep alphanumeric and basic sentence markers
    # Professional RAG benefits from keeping periods and commas for sentence boundaries
    text = re.sub(r"[^a-z0-9\s\.,!\?]", " ", text)

    # 6. Normalize punctuation (remove multiples)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    
    # 7. Final whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_products(df: pd.DataFrame, allowed_labels: List[str] = None) -> pd.DataFrame:
    """
    Keep only records for specified products using exact matching.
    
    Args:
        df: Input DataFrame
        allowed_labels: List of exact labels to keep. Defaults to config.REQUIRED_PRODUCTS.
        
    Returns:
        Filtered DataFrame
    """
    if allowed_labels is None:
        allowed_labels = config.REQUIRED_PRODUCTS
        
    df = df.copy()
    # Use exact membership check for professional precision instead of contains()
    filtered = df[df["Product"].isin(allowed_labels)].copy()
    
    print(f"[OK] Filtered products: kept {len(filtered):,} rows from the target labels.")
    return filtered


def standardize_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw product labels to standardized target categories using config.PRODUCT_MAP.
    """
    # Create an inverse mapping for fast lookup
    inv_map = {label: category for category, labels in config.PRODUCT_MAP.items() for label in labels}
    
    df = df.copy()
    df["Product"] = df["Product"].map(inv_map).fillna(df["Product"])
    
    print(f"[OK] Standardized products into categories: {list(config.PRODUCT_MAP.keys())}")
    return df


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
    print(f"[OK] Dropped {before - after:,} rows with empty narratives; {after:,} remain")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full preprocessing according to requirements:
    1. Filter products (exact labels)
    2. Standardize products (mapping variants to categories)
    3. Drop empty narratives
    4. Drop PII
    5. Clean narratives
    """
    print("Starting professional preprocessing pipeline...")
    
    # 1. Filter products
    df = filter_products(df)
    
    # 2. Standardize products
    df = standardize_products(df)
    
    # 3. Drop empty narratives
    df = drop_empty_narratives(df)
    
    # 4. Drop PII
    df = drop_pii_columns(df)
    
    # 5. Clean narratives
    df["clean_narrative"] = df["Consumer complaint narrative"].apply(clean_complaint_text)
    print("[OK] Cleaned narratives")
    
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
    
    print(f"[OK] Created stratified sample: {len(sample_df):,} rows")
    
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