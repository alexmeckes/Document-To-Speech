import numpy as np

from document_to_podcast.inference.model_loaders import TTSModel, TTS_INFERENCE
from document_to_podcast.types import SpeechParams


def text_to_speech(
    input_text: str, 
    model: TTSModel, 
    voice_profile: str,
    speech_params: SpeechParams = SpeechParams()
) -> np.ndarray:
    """
    Generate speech from text using a TTS model.

    Args:
        input_text (str): The text to convert to speech.
        model (TTSModel): The TTS model to use.
        voice_profile (str): The voice profile to use for the speech.
        speech_params (SpeechParams): Parameters for speech generation (speed, pitch, volume)
    Returns:
        np.ndarray: The waveform of the speech as a 2D numpy array
    """
    return TTS_INFERENCE[model.model_id](
        input_text, 
        model.model, 
        voice_profile,
        speech_params
    )
