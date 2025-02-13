from llama_cpp import Llama
import logging
from typing import Optional

from ..config import Config, DEFAULT_PROMPT

logger = logging.getLogger(__name__)

async def optimize_text_local(text: str, config: Config) -> str:
    """
    Optimize text using local LLM.
    
    Args:
        text: Input text to optimize
        config: Configuration object
        
    Returns:
        Optimized text string
    """
    model = Llama(
        model_path=config.text_model,
        n_ctx=4096,
        n_threads=8
    )
    
    messages = [
        {"role": "system", "content": DEFAULT_PROMPT},
        {"role": "user", "content": text}
    ]
    
    response = model.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )
    
    return response["choices"][0]["message"]["content"] 