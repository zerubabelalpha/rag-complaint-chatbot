from typing import List, Dict, Any
from langchain_core.documents import Document

from . import vectorstore
from . import llm
from . import config

# For streaming support
from threading import Thread
from transformers import TextIteratorStreamer
import torch


class RAGPipeline:
    """
    Main pipeline for the Retrieval-Augmented Generation chatbot.
    
    This class orchestrates:
    1. Loading the vector database (FAISS)
    2. Loading the Large Language Model 
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

    def run_stream(self, query: str, k: int = 5):
        """
        Execute RAG flow with streaming response (generator).
        
        Yields:
            str: Tokens as they are generated
            
        Returns:
            dict: Final metadata (at the end of generation, or accessible via side-channel)
            NOTE: In a generator, 'return' value is captured by StopIteration, 
                  so usage typically involves yielding tokens and then yielding a final object 
                  or handling metadata separately.
                  Here we will yield tokens, and the caller can retrieve sources from 
                  a separate method or we yield a special object at start/end.
        """
        if not self.is_initialized:
            if not self.initialize():
                yield "Error: Pipeline initialization failed"
                return

        # 1. Retrieve
        retrieved_docs = vectorstore.search_similar(self.vector_store, query, k=k)
        
        # 2. Format
        context_str = llm.format_docs_for_context(retrieved_docs)
        prompt = llm.RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query
        )
        
        # 3. Prepare Streaming
        # Access underlying HF components
        # self.llm_engine is a HuggingFacePipeline
        # .pipeline is the transformers pipeline
        pipe = self.llm_engine.pipeline
        model = pipe.model
        tokenizer = pipe.tokenizer
        
        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation config from our config object
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.LLM_CONFIG.max_new_tokens,
            temperature=config.LLM_CONFIG.temperature,
            do_sample=config.LLM_CONFIG.temperature > 0,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 4. Run Generation in Thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 5. Yield Documents First (so UI can show them immediately)
        # We wrap them in a special dict to distinguish from text
        yield {"source_documents": retrieved_docs}
        
        # 6. Yield tokens
        for token in streamer:
            yield token
        
        # Join thread (good practice)
        thread.join()


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