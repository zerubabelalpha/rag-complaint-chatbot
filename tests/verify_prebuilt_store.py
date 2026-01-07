from src import vectorstore, config
from pathlib import Path

def test_loading():
    print("Testing vector store loading...")
    try:
        # This should trigger the build from parquet if vector_store/faiss/index.faiss doesn't exist
        vs = vectorstore.load_vector_store()
        print(f"Successfully loaded vector store with {vs.index.ntotal} documents.")
        
        query = "I have a problem with my credit card billing"
        print(f"\nSearching for: '{query}'")
        results = vectorstore.search_similar(vs, query, k=3)
        
        vectorstore.print_search_results(results, query)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
