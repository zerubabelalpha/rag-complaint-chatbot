from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import numpy as np

from . import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the embedding model.
    
    We use sentence-transformers/all-MiniLM-L6-v2:
    - Small and fast (~80MB)
    - Good quality embeddings
    - 384-dimensional vectors
    - Works well on CPU
    
    Returns:
        HuggingFaceEmbeddings object ready to use
        
    Note:
        First call downloads the model (~80MB) to the models/hf/ folder.
        Subsequent calls load from cache (fast).
    """
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(f"  (First run will download ~80MB to {config.MODELS_DIR})")
    
    # Create embeddings object
    # model_kwargs: passed to the underlying model
    # encode_kwargs: passed when encoding text
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Force CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
        cache_folder=str(config.MODELS_DIR),  # Cache in our project folder
    )
    
    print("[OK] Embedding model loaded")
    return embeddings


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[Path] = None
) -> FAISS:
    """
    Create a new FAISS vector store from documents.
    
    This function:
    1. Loads the embedding model
    2. Embeds all documents (converts text to vectors)
    3. Stores vectors in FAISS
    4. Persists to disk for later use
    
    Args:
        documents: List of LangChain Documents to embed
        persist_directory: Where to save the database (default from config)
        
    Returns:
        FAISS vector store object
        
    Example:
        >>> chunks = chunk_documents(docs)
        >>> vectorstore = create_vector_store(chunks)
        >>> # Now you can search!
    """
    if persist_directory is None:
        persist_directory = config.VECTOR_STORE_DIR
    
    persist_directory = Path(persist_directory)
    
    # Create directory if needed
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating vector store with {len(documents):,} documents...")
    print(f"  Persist directory: {persist_directory}")
    
    # Get embedding model
    embeddings = get_embedding_model()
    
    # Create FAISS vector store
    # This embeds all documents and stores them in an in-memory FAISS index.
    # We'll then persist to disk with save_local().
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Persist FAISS index and docstore to disk
    # This creates files like:
    # - index.faiss
    # - index.pkl
    vectorstore.save_local(str(persist_directory))

    # Helpful info
    try:
        n_total = int(vectorstore.index.ntotal)
        print(f"[OK] FAISS index built (ntotal={n_total:,})")
    except Exception:
        print("[OK] FAISS index built")

    print(f"[OK] Vector store persisted to {persist_directory}")
    return vectorstore


def load_vector_store(
    persist_directory: Optional[Path] = None,
    use_prebuilt: bool = True,
    force_rebuild: bool = False
) -> FAISS:
    """
    Load or build the FAISS vector store.
    
    1. First, checks if a FAISS index exists in persist_directory.
    2. If not, and use_prebuilt=True, builds it from the parquet file.
    3. Otherwise, raises FileNotFoundError.
    
    Args:
        persist_directory: Where the FAISS files are (index.faiss, etc.)
        use_prebuilt: If True, build from data/complaint_embeddings.parquet if FAISS missing
        force_rebuild: If True, ignore existing index and rebuild from parquet
        
    Returns:
        FAISS vector store object
    """
    if persist_directory is None:
        persist_directory = config.VECTOR_STORE_DIR
    
    persist_directory = Path(persist_directory)
    faiss_index_path = persist_directory / "index.faiss"
    
    # Get embedding model (needed for both loading and querying)
    embeddings = get_embedding_model()

    # Step A: Try loading existing local FAISS index
    if faiss_index_path.exists() and not force_rebuild:
        print(f"Loading existing FAISS index from {persist_directory}...")
        vectorstore = FAISS.load_local(
            str(persist_directory),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[OK] Vector store loaded (ntotal={vectorstore.index.ntotal:,})")
        return vectorstore

    # Step B: If not found, build from pre-built parquet
    if use_prebuilt and config.PREBUILT_EMBEDDINGS_PATH.exists():
        print(f"FAISS index not found at {persist_directory}")
        print(f"Building index from pre-built source: {config.PREBUILT_EMBEDDINGS_PATH}")
        vectorstore = build_vector_store_from_parquet(
            config.PREBUILT_EMBEDDINGS_PATH,
            embeddings
        )
        
        # Persist it for faster loading next time
        print(f"Saving built index to {persist_directory} for future use...")
        persist_directory.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(persist_directory))
        return vectorstore
        
    # Step C: Give up
    raise FileNotFoundError(
        f"Vector store not found at {persist_directory} "
        f"and pre-built parquet missing at {config.PREBUILT_EMBEDDINGS_PATH}."
    )


def build_vector_store_from_parquet(
    parquet_path: Path,
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 50000
) -> FAISS:
    """
    Build a FAISS vector store from the pre-built embeddings parquet file
    using memory-efficient batch processing.
    
    Args:
        parquet_path: Path to complaint_embeddings.parquet
        embeddings: The embedding model object
        batch_size: Number of records to process at once
        
    Returns:
        Initialized FAISS vector store
    """
    import pyarrow.parquet as pq
    
    print(f"Opening parquet file: {parquet_path}...")
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"Total rows to process: {total_rows:,}")
    
    vectorstore = None
    rows_processed = 0
    
    # Iterate through the parquet file in batches
    for batch in pf.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        
        texts = df_batch['document'].tolist()
        metadatas = df_batch['metadata'].tolist()
        
        # Convert embeddings to float32 numpy array
        # Ensure we handle potential None or missing embeddings
        current_embeddings = np.stack(df_batch['embedding'].values).astype('float32')
        
        # Create text-embedding pairs
        text_embeddings = list(zip(texts, current_embeddings))
        
        if vectorstore is None:
            # Initialize FAISS with the first batch
            print(f"Initializing FAISS with first batch of {len(df_batch):,}...")
            vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=embeddings,
                metadatas=metadatas
            )
        else:
            # Add subsequent batches
            vectorstore.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=metadatas
            )
        
        rows_processed += len(df_batch)
        print(f"  Progress: {rows_processed:,} / {total_rows:,} ({rows_processed/total_rows:.1%})")
        
        # Explicitly clear batch data to free memory
        del df_batch
        del texts
        del metadatas
        del current_embeddings
        del text_embeddings
    
    print(f"[OK] Built FAISS index with {vectorstore.index.ntotal:,} vectors.")
    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = None):
    """
    Create a retriever from the vector store.
    
    A retriever is a simple interface for searching:
    - Input: query string
    - Output: list of relevant documents
    
    Args:
        vectorstore: FAISS vector store
        k: Number of documents to retrieve (default from config)
        
    Returns:
        LangChain retriever object
        
    Example:
        >>> retriever = get_retriever(vectorstore, k=5)
        >>> docs = retriever.invoke("refund request")
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
    print(f"[OK] Created retriever (k={k})")
    return retriever


def search_similar(
    vectorstore: FAISS,
    query: str,
    k: int = None
) -> List[Document]:
    """
    Search for documents similar to a query.
    
    This is a convenience function for quick searches.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query (natural language)
        k: Number of results to return
        
    Returns:
        List of similar Documents with metadata
        
    Example:
        >>> results = search_similar(vectorstore, "device not turning on", k=3)
        >>> for doc in results:
        ...     print(doc.metadata['ticket_id'], doc.page_content[:100])
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    results = vectorstore.similarity_search(query, k=k)
    return results


def search_with_scores(
    vectorstore: FAISS,
    query: str,
    k: int = None
) -> List[tuple]:
    """
    Search for documents and return similarity scores.
    
    Scores help you understand how relevant each result is.
    Lower score = more similar (for distance-based metrics).
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results
        
    Returns:
        List of (Document, score) tuples
        
    Example:
        >>> results = search_with_scores(vectorstore, "billing", k=3)
        >>> for doc, score in results:
        ...     print(f"Score: {score:.3f} - {doc.metadata['ticket_id']}")
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def print_search_results(results: List[Document], query: str) -> None:
    """
    Pretty print search results.
    
    Args:
        results: List of Documents from search
        query: The original query (for display)
    """
    print("=" * 70)
    print(f"SEARCH RESULTS for: '{query}'")
    print("=" * 70)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Complaint ID: {doc.metadata.get('complaint_id', 'N/A')}")
        print(f"Product: {doc.metadata.get('product', 'N/A')}")
        print(f"Category: {doc.metadata.get('product_category', 'N/A')}")
        print(f"Issue: {doc.metadata.get('issue', 'N/A')}")
        print(f"Company: {doc.metadata.get('company', 'N/A')}")
        print(f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}/{doc.metadata.get('total_chunks', 'N/A')}")
        print(f"Content preview:\n{doc.page_content[:300]}...")
    
    print("\n" + "=" * 70)