from typing import Optional, List, Any

import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.documents import Document
from openai import OpenAI

from . import config

class OpenAIEngine:
    """Wrapper for OpenAI-compatible API to match the pipeline interface."""
    def __init__(self, api_key: str, base_url: str, model_name: str, temperature: float, max_tokens: int):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        """Synchronous invocation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content or "_(No response)_"
        except Exception as e:
            # Provide detailed error information for debugging
            error_type = type(e).__name__
            error_msg = str(e)
            return f"❌ API Error ({error_type}): {error_msg}\n\nPlease check:\n1. API key is valid\n2. Base URL is correct\n3. Model name is supported\n4. Network connection is stable"

def get_llm(

    provider: str = None,
    model_name: str = None,
    max_new_tokens: int = None,
    temperature: float = None
) -> Any:
    """
    Load the specified LLM (Local or OpenAI).
    """
    provider = provider or config.LLM_CONFIG.provider
    model_name = model_name or (config.LLM_CONFIG.openai_model if provider == "openai" else config.LLM_CONFIG.model_name)
    max_new_tokens = max_new_tokens or config.LLM_CONFIG.max_new_tokens
    temperature = temperature or config.LLM_CONFIG.temperature

    if provider == "openai":
        print(f"Initializing OpenAI-compatible LLM: {model_name}")
        return OpenAIEngine(
            api_key=config.LLM_CONFIG.openai_api_key,
            base_url=config.LLM_CONFIG.openai_base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_new_tokens
        )

    print(f"Loading Local LLM: {model_name}")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
    )
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    # Create a custom wrapper that uses model.generate() directly
    # This avoids pipeline task type issues with transformers 5.1.0
    class LocalLLMEngine:
        """Wrapper for local seq2seq model to match the pipeline interface."""
        def __init__(self, model, tokenizer, max_new_tokens, temperature):
            self.model = model
            self.tokenizer = tokenizer
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.do_sample = temperature > 0
        
        def invoke(self, prompt: str) -> str:
            """Generate response using model.generate()."""
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    top_p=0.95 if self.do_sample else None,
                )
            
            # Decode and return
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    
    return LocalLLMEngine(model, tokenizer, max_new_tokens, temperature)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

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
    Format retrieved documents for the prompt.
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        context_parts.append(f"Content: {content}")
    
    return "\n\n".join(context_parts)
