from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import preprocess_data

sns.set_style("whitegrid")


def product_distribution(df: pd.DataFrame, product_col: str = "Product") -> pd.Series:
    """Return counts of complaints per product."""
    return df[product_col].value_counts()


def plot_product_distribution(df: pd.DataFrame, product_col: str = "Product", figsize=(12, 6)):
    counts = product_distribution(df, product_col=product_col)
    plt.figure(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.xlabel("Number of complaints")
    plt.ylabel("Product")
    plt.title("Distribution of complaints across Products")
    plt.tight_layout()


def narrative_presence_analysis(df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative") -> Dict:
    """Identify the number of complaints with and without narratives."""
    total = len(df)
    has_narrative = df[narrative_col].notna() & (df[narrative_col].astype(str).str.strip() != "")
    with_n = int(has_narrative.sum())
    without_n = total - with_n
    return {
        "total": total,
        "with_narrative": with_n,
        "without_narrative": without_n,
        "percentage_with": (with_n / total) * 100 if total > 0 else 0
    }


def narrative_length_analysis(df: pd.DataFrame, narrative_col: str = "clean_narrative") -> pd.Series:
    """Calculate the length (word count) of the Consumer complaint narrative."""
    if narrative_col not in df.columns:
        # Fallback to calculating it if not present
        return df["Consumer complaint narrative"].fillna("").apply(lambda x: len(str(x).split()))
    
    return df[narrative_col].apply(lambda x: len(str(x).split()))


def plot_narrative_length_distribution(df: pd.DataFrame, narrative_col: str = "clean_narrative", bins=50, figsize=(12, 6)):
    lengths = narrative_length_analysis(df, narrative_col=narrative_col)
    plt.figure(figsize=figsize)
    sns.histplot(lengths, bins=bins, kde=True, color="skyblue")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Consumer Complaint Narrative Lengths")
    plt.tight_layout()
    
    print(f"Narrative Length Stats:\n{lengths.describe()}")
    print(f"\nVery short narratives (< 5 words): {(lengths < 5).sum()}")
    print(f"Very long narratives (> 500 words): {(lengths > 500).sum()}")


def run_full_eda_pipeline(df: pd.DataFrame):
    """Run all EDA steps on the raw dataframe."""
    print("--- Product Distribution ---")
    print(product_distribution(df))
    plot_product_distribution(df)
    plt.show()

    print("\n--- Narrative Presence ---")
    presence = narrative_presence_analysis(df)
    print(presence)

    print("\n--- Narrative Length Distribution (after cleaning) ---")
    # Preprocess a sample or the whole thing for length analysis
    df_clean = preprocess_data(df)
    plot_narrative_length_distribution(df_clean)
    plt.show()
    
    return df_clean
