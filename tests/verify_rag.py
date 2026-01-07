import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_pipline import RAGPipeline

def verify_pipeline():
    print("--- RAG Pipeline Verification ---")
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    if not pipeline.initialize():
        print("❌ Pipeline initialization failed")
        return
    
    # Define test query
    test_query = "What are the common issues reported for credit cards related to billing?"
    
    # Run pipeline
    print(f"\nRunning test query: {test_query}")
    result = pipeline.run(test_query, k=3)
    
    if "error" in result:
        print(f"❌ Pipeline error: {result['error']}")
        return
        
    # Check results
    print("\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Sources retrieved: {len(result['source_documents'])}")
    
    if len(result['source_documents']) > 0:
        print("✅ Pipeline successfully retrieved sources and generated an answer.")
    else:
        print("⚠ No sources retrieved. Check index content.")

if __name__ == "__main__":
    verify_pipeline()
