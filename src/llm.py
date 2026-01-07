from typing import Optional
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from . import config


def get_llm(
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.1
) -> HuggingFacePipeline:
    """
    Load the LLM for text generation.
    
    We use google/flan-t5-small:
    - Instruction-tuned T5 model
    - ~300MB download
    - Runs on CPU
    - Good for demos and learning
    
    Args:
        model_name: HuggingFace model name (default from config)
        max_new_tokens: Maximum tokens to generate (default 256)
        temperature: Randomness (0=deterministic, 1=creative)
        
    Returns:
        LangChain-compatible LLM object
        
    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is RAG?")
        >>> print(response)
    """
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    
    print(f"Loading LLM: {model_name}")
    print(f"  (First run will download ~300MB to {config.MODELS_DIR})")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    
    # Load tokenizer
    # The tokenizer converts text to tokens (numbers) and back
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
    )
    
    # Load model
    # AutoModelForSeq2SeqLM is for encoder-decoder models like T5
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
    )
    
    # Create a HuggingFace pipeline
    # This wraps the model for easy text generation
    pipe = pipeline(
        "text2text-generation",  # Task type for T5
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,  # Only sample if temperature > 0
        device=-1,  # -1 = CPU, 0 = first GPU
        truncation=True, # Ensure we don't crash on long inputs
    )
    
    # Wrap in LangChain's HuggingFacePipeline for compatibility
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("[OK] LLM loaded and ready")
    return llm


def test_llm(llm) -> str:
    """
    Quick test to verify the LLM is working.
    
    Args:
        llm: LangChain LLM object
        
    Returns:
        Generated response
    """
    test_prompt = "What is customer support? Answer in one sentence."
    
    print("Testing LLM with prompt:", test_prompt)
    response = llm.invoke(test_prompt)
    print(f"Response: {response}")
    
    return response


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# The RAG prompt template
# This tells the LLM how to behave and what format to use
RAG_PROMPT_TEMPLATE = """You are a professional financial analyst. Use the provided context to answer the user's question accurately. 

Guidelines:
1. Use ONLY the provided context to answer the question.
2. If the answer is not in the context, say "I don't have enough information to answer this based on the available complaints."
3. Maintain a helpful and professional tone.
4. If multiple pieces of information are relevant, summarize them clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_docs_for_context(docs: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.
    
    Args:
        docs: List of LangChain Documents
        
    Returns:
        Formatted string with all document contents
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Extract metadata if available for better context
        complaint_id = doc.metadata.get("complaint_id", "Unknown ID")
        content = doc.page_content.strip()
        context_parts.append(f"--- Document {i} (Complaint {complaint_id}) ---\n{content}")
    
    return "\n\n".join(context_parts)

    
    
