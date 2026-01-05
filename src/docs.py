
import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document


def row_to_document(row: pd.Series) -> Document:
    """
    Convert a single DataFrame row to a LangChain Document.
    
    Args:
        row: A pandas Series (one row from DataFrame)
        
    Returns:
        LangChain Document object
    """
    # The text content that will be embedded
    # We use 'clean_narrative' created during preprocessing
    page_content = str(row.get("clean_narrative", ""))
    
    # Metadata: information to keep for filtering and context
    metadata = {
        "complaint_id": row.get("Complaint ID", ""),
        "product": row.get("Product", ""),
        "sub_product": row.get("Sub-product", ""),
        "issue": row.get("Issue", ""),
        "sub_issue": row.get("Sub-issue", ""),
        "company": row.get("Company", ""),
        "state": row.get("State", ""),
        "date_received": str(row.get("Date received", "")),
        "timely_response": row.get("Timely response?", ""),
        "consumer_disputed": row.get("Consumer disputed?", ""),
    }
    
    # Clean up metadata: convert NaN to None, ensure strings
    cleaned_metadata = {}
    for key, value in metadata.items():
        if pd.isna(value):
            cleaned_metadata[key] = None
        elif isinstance(value, (int, float)):
            cleaned_metadata[key] = value
        else:
            cleaned_metadata[key] = str(value)
    
    return Document(page_content=page_content, metadata=cleaned_metadata)


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert entire DataFrame to list of LangChain Documents.
    
    This is the main function to prepare data for the vector store.
    
    Args:
        df: DataFrame with 'clean_narrative' column and metadata columns
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for idx, row in df.iterrows():
        doc = row_to_document(row)
        
        # Skip empty documents
        if doc.page_content.strip():
            documents.append(doc)
    
    print(f"✓ Converted {len(documents):,} rows to LangChain Documents")
    
    # Show sample
    if documents:
        print(f"  Sample metadata keys: {list(documents[0].metadata.keys())}")
    
    return documents


def print_document_sample(doc: Document, max_content_length: int = 200) -> None:
    """
    Pretty print a Document for inspection.
    """
    print("=" * 60)
    print("DOCUMENT SAMPLE")
    print("=" * 60)
    
    # Show content (truncated)
    content = doc.page_content
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    print(f"Content:\n{content}")
    
    print("-" * 60)
    print("Metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print("=" * 60)
