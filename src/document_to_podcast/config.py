from pathlib import Path
from typing import Literal, Dict, Optional
from typing_extensions import Annotated
from enum import Enum

from pydantic import BaseModel, FilePath
from pydantic.functional_validators import AfterValidator

from document_to_podcast.inference.model_loaders import TTS_LOADERS
from document_to_podcast.types import SpeechParams
from document_to_podcast.preprocessing import DATA_LOADERS


DEFAULT_PROMPT = """
You are an expert text optimizer for speech synthesis. Your task is to enhance the given text to make it more suitable for text-to-speech conversion while preserving its meaning and intent.

Instructions:
- Remove all HTML tags and CSS styling (e.g., convert <p> tags to line breaks)
- Extract meaningful text from HTML elements (e.g., alt text from images, link text)
- Convert HTML entities to their spoken equivalents (e.g., "&amp;" to "and")
- Fix any grammatical issues
- Improve sentence structure for natural speech flow
- Add proper punctuation for better pacing
- Break down complex sentences into simpler ones
- Format numbers, dates, and special characters for speech (e.g., "2023" -> "twenty twenty-three")
- Expand abbreviations and acronyms when appropriate
- Add natural pauses through punctuation
- Maintain the original meaning and key information
- Keep the text professional and clear
- For code blocks or command-line instructions, describe them conversationally
- For URLs and file paths, read them naturally with appropriate pauses

Return the optimized text in a clean format, preserving paragraph structure.
"""

def validate_input_file(value):
    if Path(value).suffix not in DATA_LOADERS:
        raise ValueError(
            f"input_file extension must be one of {list(DATA_LOADERS.keys())}"
        )
    return value

def validate_text_to_text_model(value):
    # Could add additional validation for specific LLM models if needed
    return value

def validate_text_to_speech_model(value):
    if value not in TTS_LOADERS:
        raise ValueError(
            f"Model {value} is missing a loading function. Please define it under model_loaders.py"
        )
    return value

class TextOptimizationModel(str, Enum):
    LOCAL_LLM = "local_llm"
    OPENROUTER = "openrouter"

class Config(BaseModel):
    input_file: Annotated[FilePath, AfterValidator(validate_input_file)]
    output_file: str
    optimize_text: bool = True
    text_optimization_model: TextOptimizationModel = TextOptimizationModel.LOCAL_LLM
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-3-opus"  # Default to Claude 3 Opus
    text_model: Annotated[str, AfterValidator(validate_text_to_text_model)] = "bartowski/Qwen2.5-7B-Instruct-GGUF"
    tts_model: Annotated[str, AfterValidator(validate_text_to_speech_model)] = "hexgrad/Kokoro-82M"
    voice_profile: str = "en_neutral"
    chunk_size: int = 1000
    optimization_level: Literal["basic", "moderate", "aggressive"] = "moderate"
    speech_params: SpeechParams = SpeechParams()
