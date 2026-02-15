from typing import List, Dict, Any
from langchain_core.documents import Document

from . import vectorstore
from . import llm
from . import config


class RAGPipeline:
    """
    Main pipeline for the Retrieval-Augmented Generation chatbot.
    
    This class orchestrates:
    1. Loading the vector database (FAISS)
    2. Loading the Large Language Model (FLAN-T5)
    3. Processing user queries (Retrieve -> Augment -> Generate)
    """
    
    def __init__(self):
        self.vector_store = None
        self.llm_engine = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Load necessary models and data."""
        print("\n--- Initializing RAG Pipeline ---")
        
        # 1. Load Vector Store
        print(f"Attempting to load vector store from: {config.VECTOR_STORE_DIR}")
        try:
            self.vector_store = vectorstore.load_vector_store()
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print(f"Does path exist? {config.VECTOR_STORE_DIR.exists()}")
            print("Hint: Run indexing first.")
            return False
            
        # 2. Load LLM
        self.llm_engine = llm.get_llm(
            temperature=config.LLM_CONFIG.temperature,
            max_new_tokens=config.LLM_CONFIG.max_new_tokens
        )
        
        self.is_initialized = True
        print("--- Pipeline Ready ---\n")
        return True

    def run(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Execute the full RAG flow for a user query.
        """
        if not self.is_initialized:
            if not self.initialize():
                return {"error": "Pipeline initialization failed"}

        print(f"Processing Query: '{query}'")
        
        # Step 1: Retrieve relevant documents
        # We use similarity search (Top-K)
        retrieved_docs = vectorstore.search_similar(self.vector_store, query, k=k)
        
        # Step 2: Augment - Format docs into a context string
        context_str = llm.format_docs_for_context(retrieved_docs)
        
        # Step 3: Generate - Fill the template and call LLM
        prompt = llm.RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query
        )
        
        print(f"Generating answer based on {len(retrieved_docs)} sources...")
        answer = self.llm_engine.invoke(prompt)
        
        # Step 4: Return result and source metadata
        return {
            "query": query,
            "answer": answer.strip(),
            "source_documents": retrieved_docs
        }


def main() -> None:
    """Simple CLI entry point for testing."""
    pipeline = RAGPipeline()
    if not pipeline.initialize():
        return
        
    while True:
        user_query = input("\nEnter your question (or 'exit'): ")
        if user_query.lower() in ["exit", "quit"]:
            break
            
        result = pipeline.run(user_query)
        
        print("\n" + "="*50)
        print("ANSWER:")
        print(result["answer"])
        print("="*50)
        
        print("\nSOURCES:")
        for i, doc in enumerate(result["source_documents"], 1):
            cid = doc.metadata.get("complaint_id", "N/A")
            company = doc.metadata.get("company", "N/A")
            print(f"{i}. [ID: {cid}] {company}")


if __name__ == "__main__":
    main()