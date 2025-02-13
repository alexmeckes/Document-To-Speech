import httpx
import json
from typing import Optional, List, Dict

from ..config import Config, DEFAULT_PROMPT

async def get_available_models(api_key: str) -> List[Dict]:
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key: OpenRouter API key
        
    Returns:
        List of model information dictionaries
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/your-repo/document-to-podcast",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.text}")
            
        data = response.json()
        print("Raw API Response:", data)  # Debug print
        
        if not isinstance(data, dict) or "data" not in data:
            raise Exception("Unexpected API response format")
            
        models = data["data"]
        # Filter for chat models and sort by pricing
        chat_models = []
        for model in models:
            if isinstance(model, dict):
                # Extract relevant information
                model_info = {
                    "id": model.get("id", ""),
                    "name": model.get("name", ""),
                    "context_length": model.get("context_length", 0),
                    "pricing": model.get("pricing", {})  # Get the full pricing object
                }
                print(f"Model pricing for {model_info['name']}:", model.get("pricing", {}))  # Debug print
                if model_info["context_length"] > 0:  # Only include models that can process text
                    chat_models.append(model_info)
                    
        return sorted(chat_models, key=lambda x: float(x["pricing"].get("input", 0)))

async def optimize_text_openrouter(text: str, config: Config) -> str:
    """
    Optimize text using OpenRouter API.
    
    Args:
        text: Input text to optimize
        config: Configuration object containing OpenRouter settings
        
    Returns:
        Optimized text string
    """
    if not config.openrouter_api_key:
        raise ValueError("OpenRouter API key is required for OpenRouter text optimization")
        
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/your-repo/document-to-podcast",  # Update with your repo
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": DEFAULT_PROMPT},
        {"role": "user", "content": text}
    ]
    
    data = {
        "model": config.openrouter_model,
        "messages": messages
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.text}")
            
        result = response.json()
        return result["choices"][0]["message"]["content"] 