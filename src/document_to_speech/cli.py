import re
from pathlib import Path

import soundfile as sf
import yaml
from fire import Fire
from loguru import logger

from document_to_speech.config import (
    Config,
    DEFAULT_PROMPT,
)
from document_to_speech.inference.model_loaders import (
    load_llama_cpp_model,
    load_tts_model,
)
from document_to_speech.inference.text_to_speech import text_to_speech
from document_to_speech.inference.text_optimizer import optimize_text
from document_to_speech.preprocessing import DATA_CLEANERS, DATA_LOADERS


@logger.catch(reraise=True)
async def document_to_speech(
    input_file: str | None = None,
    output_file: str | None = None,
    text_model: str = "bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q8_0.gguf",
    text_to_speech_model: str = "hexgrad/Kokoro-82M",
    voice_profile: str = "en_neutral",
    optimize_text: bool = True,
    chunk_size: int = 1000,
    optimization_level: str = "moderate",
    from_config: str | None = None,
):
    """
    Convert a document to speech with optional text optimization.

    Args:
        input_file (str): The path to the input file.
            Supported extensions:
                - .pdf
                - .html
                - .txt
                - .docx
                - .md

        output_file (str): The path to the output audio file.
            Will be saved as a .wav file.

        text_model (str, optional): The LLM model for text optimization.
            Defaults to Qwen2.5-7B-Instruct.

        text_to_speech_model (str, optional): The text-to-speech model_id.
            Defaults to `hexgrad/Kokoro-82M`.

        voice_profile (str, optional): The voice profile to use.
            Defaults to "en_neutral".

        optimize_text (bool, optional): Whether to optimize text using LLM.
            Defaults to True.

        chunk_size (int, optional): Size of text chunks for processing.
            Defaults to 1000.

        optimization_level (str, optional): Level of text optimization.
            One of: "basic", "moderate", "aggressive"
            Defaults to "moderate".

        from_config (str, optional): The path to the config file.
            If provided, all other arguments will be ignored.
    """
    if from_config:
        config = Config.model_validate(yaml.safe_load(Path(from_config).read_text()))
    else:
        config = Config(
            input_file=input_file,
            output_file=output_file,
            text_model=text_model,
            tts_model=text_to_speech_model,
            voice_profile=voice_profile,
            optimize_text=optimize_text,
            chunk_size=chunk_size,
            optimization_level=optimization_level,
        )

    # Create output directory if needed
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and clean document
    data_loader = DATA_LOADERS[Path(config.input_file).suffix]
    logger.info(f"Loading {config.input_file}")
    raw_text = data_loader(config.input_file)
    logger.debug(f"Loaded {len(raw_text)} characters")

    data_cleaner = DATA_CLEANERS[Path(config.input_file).suffix]
    logger.info(f"Cleaning {config.input_file}")
    clean_text = data_cleaner(raw_text)
    logger.debug(f"Cleaned {len(raw_text) - len(clean_text)} characters")
    logger.debug(f"Length of cleaned text: {len(clean_text)}")

    # Text optimization if enabled
    if config.optimize_text:
        logger.info("Loading text optimization model")
        text_model = load_llama_cpp_model(model_id=config.text_model)
        
        logger.info("Optimizing text for speech")
        processed_text = await optimize_text(
            clean_text,
            text_model,
            DEFAULT_PROMPT,
            config.chunk_size,
            config.optimization_level
        )
    else:
        processed_text = clean_text

    # Text to speech conversion
    logger.info(f"Loading {config.tts_model}")
    speech_model = load_tts_model(
        model_id=config.tts_model,
        **{"lang_code": config.voice_profile[0]}  # First character of voice profile is language code
    )

    logger.info("Generating speech")
    audio = text_to_speech(
        processed_text,
        speech_model,
        config.voice_profile,
        config.speech_params
    )

    # Save output
    logger.info(f"Saving audio to {config.output_file}")
    sf.write(
        config.output_file,
        audio,
        samplerate=speech_model.sample_rate,
    )
    
    # Save processed text for reference
    text_output = str(output_path.with_suffix('.txt'))
    logger.info(f"Saving processed text to {text_output}")
    Path(text_output).write_text(processed_text)
    
    logger.success("Done!")


def main():
    Fire(document_to_speech)
