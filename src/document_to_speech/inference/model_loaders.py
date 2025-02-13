from dataclasses import dataclass
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from kokoro import KPipeline
from llama_cpp import Llama
import numpy as np

from document_to_speech.types import SpeechParams


@dataclass
class TTSModel:
    model_id: str
    model: Any
    sample_rate: int
    custom_args: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_args is None:
            self.custom_args = {}


def load_llama_cpp_model(model_id: str) -> Llama:
    """
    Load a LLaMA model from the Hugging Face Hub.
    Args:
        model_id: Format should be 'namespace/repo_name' (e.g., 'bartowski/Qwen2.5-7B-Instruct-GGUF')
    """
    # Split the model_id into repo_id and filename if needed
    parts = model_id.split("/")
    if len(parts) > 2:
        # If model_id includes filename, extract it
        repo_id = "/".join(parts[:2])
        filename = parts[2]
    else:
        # Otherwise use default filename
        repo_id = model_id
        filename = "Qwen2.5-7B-Instruct-Q8_0.gguf"

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="models",
    )
    return Llama(model_path=model_path)


def load_kokoro_model(model_id: str, **kwargs) -> TTSModel:
    """
    Load a Kokoro model from the Hugging Face Hub.
    Args:
        model_id: The model ID (e.g., 'hexgrad/Kokoro-82M')
        **kwargs: Additional arguments, including:
            lang_code: Single letter language code ('a' for American English, 'b' for British English)
    """
    if 'lang_code' not in kwargs:
        raise ValueError("lang_code is required for Kokoro models")
    
    model = KPipeline(lang_code=kwargs['lang_code'])
    return TTSModel(
        model_id=model_id,
        model=model,
        sample_rate=24000,
        custom_args=kwargs
    )


TTS_LOADERS = {
    # To add support for your model, add it here in the format {model_id} : _loader_function
    "hexgrad/Kokoro-82M": load_kokoro_model,
}


def load_tts_model(model_id: str, **kwargs) -> TTSModel:
    """
    Load a TTS model from the Hugging Face Hub.
    """
    return TTS_LOADERS[model_id](model_id, **kwargs)


def _text_to_speech_kokoro(
    input_text: str, 
    model: KPipeline, 
    voice_profile: str,
    speech_params: SpeechParams = SpeechParams()
) -> np.ndarray:
    """
    TTS generation function for the Kokoro model
    Args:
        input_text (str): The text to convert to speech.
        model (KPipeline): The kokoro pipeline as defined in https://github.com/hexgrad/kokoro
        voice_profile (str) : a pre-defined ID for the Kokoro models (e.g. "af_bella")
            more info here https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        speech_params (SpeechParams): Parameters for speech generation (speed, pitch, volume)

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    generator = model(
        input_text, 
        voice=voice_profile,
        speed=speech_params.speed,
    )

    # Process all chunks from the generator
    audio_chunks = []
    for _, _, audio in generator:  # returns graphemes/text, phonemes, audio for each chunk
        if speech_params.volume != 1.0:
            audio = audio * speech_params.volume
        audio_chunks.append(audio)
    
    # Concatenate all audio chunks
    if audio_chunks:
        return np.concatenate(audio_chunks)
    else:
        return np.array([])  # Return empty array if no audio was generated


TTS_INFERENCE = {
    # To add support for your model, add it here in the format {model_id} : _inference_function
    "hexgrad/Kokoro-82M": _text_to_speech_kokoro,
}
