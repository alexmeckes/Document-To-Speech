import logging
from typing import List
from llama_cpp import Llama

from ..config import Config, TextOptimizationModel
from .openrouter_optimizer import optimize_text_openrouter
from .local_llm_optimizer import optimize_text_local  # We'll create this from the existing code

logger = logging.getLogger(__name__)

def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters,
    trying to break at sentence boundaries.
    """
    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    sentences = text.split(". ")
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            # Add current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def optimize_text_chunk(
    chunk: str,
    model: Llama,
    system_prompt: str,
    optimization_level: str = "moderate"
) -> str:
    """
    Optimize a chunk of text for speech synthesis using the LLM.
    """
    # Adjust prompt based on optimization level
    level_instructions = {
        "basic": "Focus only on critical fixes for grammar and punctuation.",
        "moderate": "Apply all standard optimizations as listed in the instructions.",
        "aggressive": "Apply all optimizations and make significant restructuring for maximum speech clarity."
    }
    
    full_prompt = f"{system_prompt}\nOptimization Level: {level_instructions[optimization_level]}\n\nText to optimize:\n{chunk}"
    
    try:
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
            stream=False,
        )
        
        optimized_text = response["choices"][0]["message"]["content"]
        return optimized_text.strip()
    
    except Exception as e:
        logger.error(f"Error during text optimization: {e}")
        # Return original text if optimization fails
        return chunk

async def optimize_text(text: str, config: Config) -> str:
    """
    Optimize text using either local LLM or OpenRouter based on configuration.
    
    Args:
        text: Input text to optimize
        config: Configuration object
        
    Returns:
        Optimized text string
    """
    if not config.optimize_text:
        return text
        
    # Split text into chunks for processing
    chunks = split_text_into_chunks(text, config.chunk_size)
    optimized_chunks: List[str] = []
    
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Optimizing chunk {i}/{len(chunks)}")
        
        try:
            if config.text_optimization_model == TextOptimizationModel.LOCAL_LLM:
                optimized_chunk = await optimize_text_local(chunk, config)
            else:  # OpenRouter
                optimized_chunk = await optimize_text_openrouter(chunk, config)
            optimized_chunks.append(optimized_chunk)
        except Exception as e:
            logger.error(f"Error optimizing chunk {i}: {str(e)}")
            # On error, keep original chunk
            optimized_chunks.append(chunk)
    
    return "\n".join(optimized_chunks)

def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks while preserving paragraph boundaries."""
    if not text:
        return []
        
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        
        if current_size + paragraph_size > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(paragraph)
        current_size += paragraph_size
        
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks 