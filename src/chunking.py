from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified parameters.
    
    Args:
        chunk_size: Max characters per chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        
    Returns:
        Configured RecursiveCharacterTextSplitter
        
    Example:
        >>> splitter = create_text_splitter(chunk_size=300, chunk_overlap=30)
    """
    # Use config defaults if not specified
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    # Create the splitter
    # separators: Try to split on these in order (paragraph, newline, sentence, word, char)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count
        separators=["\n\n", "\n", ". ", " ", ""],  # Natural boundaries
        is_separator_regex=False,
    )
    
    print(f"✓ Created text splitter (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return splitter


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Each chunk inherits the metadata from its parent document.
    We also add chunk-specific metadata (chunk_index).
    
    Args:
        documents: List of LangChain Documents to chunk
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Documents (usually more than input)
        
    Example:
        >>> docs = dataframe_to_documents(df)  # 8000 docs
        >>> chunks = chunk_documents(docs)      # Maybe 12000 chunks
    """
    # Create splitter
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Split all documents
    # split_documents preserves metadata automatically!
    chunked_docs = splitter.split_documents(documents)
    
    # Add chunk index to metadata for traceability
    # Group by 'complaint_id' to number chunks within each record
    record_chunk_counts = {}
    for doc in chunked_docs:
        record_id = doc.metadata.get("complaint_id", "unknown")
        
        # Initialize or increment chunk counter for this record
        if record_id not in record_chunk_counts:
            record_chunk_counts[record_id] = 0
        
        doc.metadata["chunk_index"] = record_chunk_counts[record_id]
        record_chunk_counts[record_id] += 1
    
    # Report statistics
    original_count = len(documents)
    chunked_count = len(chunked_docs)
    expansion_ratio = chunked_count / original_count if original_count > 0 else 0
    
    print(f"✓ Chunking complete:")
    print(f"  Original documents: {original_count:,}")
    print(f"  After chunking: {chunked_count:,}")
    print(f"  Expansion ratio: {expansion_ratio:.2f}x")
    
    return chunked_docs


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Calculate statistics about chunk sizes.
    
    Useful for understanding your chunking results.
    
    Args:
        chunks: List of chunked Documents
        
    Returns:
        Dictionary with min, max, mean, median chunk lengths
    """
    lengths = [len(doc.page_content) for doc in chunks]
    
    if not lengths:
        return {"error": "No chunks provided"}
    
    stats = {
        "total_chunks": len(chunks),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }
    
    return stats


def print_chunk_samples(chunks: List[Document], n_samples: int = 2) -> None:
    """
    Print a few sample chunks to see what they look like.
    """
    print("\n" + "=" * 50)
    print(f"SAMPLE CHUNKS (Showing {min(n_samples, len(chunks))} of {len(chunks)})")
    print("=" * 50)
    
    for i in range(min(n_samples, len(chunks))):
        doc = chunks[i]
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Content:\n{doc.page_content[:300]}...")
        if len(doc.page_content) > 300:
            print("[...]")
            
    print("\n" + "=" * 50)