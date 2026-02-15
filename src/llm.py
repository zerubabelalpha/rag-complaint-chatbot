from typing import Optional, List
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document

from . import config

def get_llm(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> HuggingFacePipeline:
    """
    Load the Microsoft Phi-3-mini model.
    
    Phi-3-mini is a high-performance 3.8B parameter model that:
    - Performs significantly better than T5 for reasoning tasks.
    - Fits on most consumer hardware (approx 4GB-8GB VRAM/RAM).
    - Uses 'text-generation' (Causal LM) architecture.
    
    Returns:
        LangChain-compatible HuggingFacePipeline
    """
    print(f"Loading Alternative LLM: {model_name}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
        trust_remote_code=True
    )
    
    # Load model
    # We use AutoModelForCausalLM for Phi-3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
        trust_remote_code=True,
        torch_dtype="auto",       # Automatically choose float16/bfloat16
        device_map="auto",        # Automatically use GPU if available
    )

    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        trust_remote_code=True,
        # Phi-3 works best when we disable the "End of sentence" token in return
        return_full_text=False, 
    )

    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# PHI-3 SPECIFIC PROMPT TEMPLATE
# =============================================================================

# Phi-3 uses a specific format with <|user|> and <|assistant|> tags
RAG_PROMPT_TEMPLATE = """<|user|>
You are a professional financial analyst. Use the provided context to answer the user's question accurately.

Guidelines:
1. Use ONLY the provided context to answer the question.
2. If the answer is not in the context, say "I don't have enough information."
3. Maintain a professional tone.

CONTEXT:
{context}

QUESTION: {question}<|end|>
<|assistant|>"""


def format_docs_for_context(docs: List[Document]) -> str:
    """
    Format retrieved documents for the Phi-3 prompt.
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        complaint_id = doc.metadata.get("complaint_id", "Unknown ID")
        content = doc.page_content.strip()
        context_parts.append(f"[Document {i} | Complaint {complaint_id}]\n{content}")
    
    return "\n\n".join(context_parts)
