import os
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Input file
RAW_DATA_PATH = RAW_DATA_DIR / "complaints.csv"

# Output file
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "filtered_complaints.csv"

# Vector store path 
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss"
PREBUILT_EMBEDDINGS_PATH = DATA_DIR / "complaint_embeddings.parquet"

# Model cache directory
MODELS_DIR = PROJECT_ROOT / "models" / "hf"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL_NAME = "google/flan-t5-small"

@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""
    model_name: str = "google/flan-t5-small"
    temperature: float = 0.1
    max_new_tokens: int = 512


@dataclass
class DisplayConfig:
    """Configuration for display and UI constants."""
    content_preview_length: int = 300
    max_content_length: int = 200
    separator_length: int = 60


# Default configuration instances
LLM_CONFIG = LLMConfig()
DISPLAY_CONFIG = DisplayConfig()

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 50


# Default chunking configuration instance
CHUNKING_CONFIG = ChunkingConfig()

# Backward compatibility - keep module-level constants
CHUNK_SIZE = CHUNKING_CONFIG.chunk_size
CHUNK_OVERLAP = CHUNKING_CONFIG.chunk_overlap

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    # Number of documents to retrieve for each query
    # - More docs = more context but slower and may include noise
    # - 3 is a safer bet for small models like FLAN-T5-small (512 token limit)
    k: int = 3


# Default retrieval configuration instance
RETRIEVAL_CONFIG = RetrievalConfig()

# Backward compatibility - keep module-level constant
RETRIEVAL_K = RETRIEVAL_CONFIG.k

# =============================================================================
# HUGGINGFACE CACHE SETUP
# =============================================================================

def setup_hf_cache() -> None:
    """
    Configure HuggingFace to cache models in our project's models/ folder.
    
    WHY THIS MATTERS:
    - By default, HF downloads models to ~/.cache/huggingface/
    - Setting HF_HOME keeps everything in our project folder
    - This makes the project more portable and self-contained
    
    CALL THIS EARLY in your code (before importing transformers/sentence_transformers)
    """
    # Create the models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set the environment variable
    # HF_HOME is the master setting that controls all HF caching
    os.environ["HF_HOME"] = str(MODELS_DIR)
    
    # Also set these for older versions of the libraries
    os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)
    
    print(f"[OK] HuggingFace cache set to: {MODELS_DIR}")


# Dataset Configuration (CFPB Complaints)
# =============================================================================

# Mapping of raw dataset product labels to standardized target categories.
# This consolidates labels from different dataset versions into professional categories.

PRODUCT_MAP = {
    "Credit card": [
        "Credit card",
        "Credit card or prepaid card",
        "Prepaid card"
    ],
    "Personal loan": [
        "Consumer Loan",
        "Payday loan",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Student loan",
        "Vehicle loan or lease"
    ],
    "Savings account": [
        "Checking or savings account",
        "Bank account or service"
    ],
    "Money transfers": [
        "Money transfers",
        "Money transfer, virtual currency, or money service",
        "Virtual currency"
    ],
    "Buy Now, Pay Later (BNPL)": [
        "Buy Now, Pay Later (BNPL)"
    ]
}

# The five target categories specified for inclusion in the RAG pipeline.
TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfers",
    "Buy Now, Pay Later (BNPL)"
]

# Flattened list of raw labels corresponding to the target categories.
# Used by the preprocessing pipeline for strict filtering.
REQUIRED_PRODUCTS = [label for cat in TARGET_PRODUCTS for label in PRODUCT_MAP.get(cat, [])]

# Columns that contain PII or are irrelevant and MUST be removed before embedding
# NEVER include these in the text that gets embedded!
PII_COLUMNS_TO_DROP = [
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Company public response"
]

# Columns to keep as metadata (not embedded, but stored for reference)
METADATA_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID"
]

# Columns that will be combined into the document text for embedding
# For this project, we primarily use the cleaned narrative
TEXT_COLUMNS_FOR_EMBEDDING = ["clean_narrative"]