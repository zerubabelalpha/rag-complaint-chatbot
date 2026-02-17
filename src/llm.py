from typing import Optional, List
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.documents import Document

from . import config

def get_llm(
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> HuggingFacePipeline:
    """
    Load the Google FLAN-T5-small model.
    
    FLAN-T5-small is an 80M parameter Seq2Seq model that:
    - Runs very fast on CPU.
    - Is suitable for summarization and simple QA.
    - Uses 'text2text-generation' architecture.
    
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
    )
    
    # Load model
    # We use AutoModelForSeq2SeqLM for T5
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    # Create the pipeline
    pipe = pipeline(
        "text-generation", # Fallback to text-generation if text2text-generation is missing
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# FLAN-T5 SPECIFIC PROMPT TEMPLATE
# =============================================================================

# T5 expects a direct instruction. No need for complex chat tags.

RAG_PROMPT_TEMPLATE = """You are a helpful financial assistant. Answer the question using ONLY the following context.

Guidelines:
1. Use ONLY the provided context to answer the question.
2. If the answer is not in the context, say "I don't have enough information to answer this based on the available complaints."
3. Maintain a helpful and professional tone.
4. If multiple pieces of information are relevant, summarize them clearly.

Context:
{context}

Question: {question}

Answer:"""


def format_docs_for_context(docs: List[Document]) -> str:
    """
    Format retrieved documents for the T5 prompt.
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        context_parts.append(f"Content: {content}")
    
    return "\n\n".join(context_parts)
