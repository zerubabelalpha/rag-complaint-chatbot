from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

from .preprocess import preprocess_data

sns.set_style("whitegrid")


def product_distribution(df: pd.DataFrame, product_col: str = "Product") -> pd.Series:
    """Return counts of complaints per product."""
    return df[product_col].value_counts()


def plot_product_distribution(df: pd.DataFrame, product_col: str = "Product", title: str = "Distribution of complaints across Products", figsize=(12, 6)):
    counts = product_distribution(df, product_col=product_col)
    plt.figure(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.xlabel("Number of complaints")
    plt.ylabel("Product")
    plt.title(title)
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


def plot_length_by_product(df: pd.DataFrame, narrative_col: str = "clean_narrative", product_col: str = "Product"):
    """Visualize narrative length across different product categories."""
    df_plot = df.copy()
    df_plot['length'] = df_plot[narrative_col].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='length', y=product_col, data=df_plot, palette="muted")
    plt.title("Narrative Length Distribution by Product Category")
    plt.xlabel("Word Count")
    plt.xscale('log')  # Use log scale for better visibility of spread
    plt.tight_layout()


def get_top_keywords(text_series: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
    """Simple keyword extraction by counting words, excluding very short ones."""
    # Basic stop words - we can expand this or use a library if available
    stop_words = {'the', 'and', 'to', 'was', 'my', 'that', 'for', 'with', 'on', 'this', 'had', 'it', 'they', 'at', 'be', 'me', 'have', 'of', 'in', 'is', 'i'}
    
    all_text = " ".join(text_series.astype(str)).lower()
    words = re.findall(r'\b\w{4,}\b', all_text) # Only words with 4+ chars
    filtered_words = [w for w in words if w not in stop_words]
    
    return Counter(filtered_words).most_common(top_n)


def plot_top_keywords(df: pd.DataFrame, text_col: str = "clean_narrative", top_n: int = 15, title: str = "Top Keywords in Complaints"):
    """Visualize most common words in narratives."""
    top_words = get_top_keywords(df[text_col], top_n=top_n)
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="rocket")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()


def plot_company_distribution(df: pd.DataFrame, top_n: int = 15):
    """Visualize distribution of complaints across top companies."""
    counts = df["Company"].value_counts().head(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index, palette="magma")
    plt.title(f"Top {top_n} Companies by Complaint Volume")
    plt.xlabel("Number of Complaints")
    plt.tight_layout()


def plot_temporal_trends(df: pd.DataFrame):
    """Visualize complaint volume over time."""
    df_temp = df.copy()
    df_temp['Date received'] = pd.to_datetime(df_temp['Date received'])
    daily_counts = df_temp.resample('M', on='Date received').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind='line', marker='o', color='teal')
    plt.title("Monthly Complaint Volume Trend")
    plt.xlabel("Date")
    plt.ylabel("Number of Complaints")
    plt.tight_layout()


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
