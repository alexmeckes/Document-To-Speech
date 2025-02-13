"""Streamlit app for converting documents to speech."""

import io
import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from document_to_speech.inference.text_to_speech import text_to_speech
from document_to_speech.preprocessing import DATA_LOADERS, DATA_CLEANERS
from document_to_speech.inference.model_loaders import (
    load_llama_cpp_model,
    load_tts_model,
)
from document_to_speech.config import DEFAULT_PROMPT, SpeechParams, TextOptimizationModel, Config
from document_to_speech.inference.text_optimizer import optimize_text
from document_to_speech.inference.openrouter_optimizer import get_available_models

# Initialize session state for OpenRouter models
if "openrouter_models" not in st.session_state:
    st.session_state.openrouter_models = []

@st.cache_resource
def load_text_to_text_model():
    return load_llama_cpp_model(
        model_id="bartowski/Qwen2.5-7B-Instruct-GGUF"
    )


@st.cache_resource
def load_text_to_speech_model(voice_profile: str):
    # Extract language code from voice profile (first character)
    lang_code = voice_profile[0]  # 'a' for American English, 'b' for British English
    return load_tts_model("hexgrad/Kokoro-82M", lang_code=lang_code)


def numpy_to_wav(audio_array: np.ndarray, sample_rate: int) -> io.BytesIO:
    """
    Convert a numpy array to audio bytes in .wav format, ready to save into a file.
    """
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_array, sample_rate, format="WAV")
    wav_io.seek(0)
    return wav_io


# Initialize session state
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "audio" not in st.session_state:
    st.session_state.audio = None
if "generate_button" not in st.session_state:
    st.session_state.generate_button = False


def gen_button_clicked():
    st.session_state.generate_button = True


sample_rate = 24000
st.title("Document To Speech Converter")

st.header("Upload a File")

uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "docx", "md"]
)

st.header("Or Enter a Website URL")
url = st.text_input("URL", placeholder="https://example.com/...")

if uploaded_file is not None or url:
    st.divider()
    st.header("Loading and Cleaning Data")
    st.divider()

    if uploaded_file:
        extension = Path(uploaded_file.name).suffix
        raw_text = DATA_LOADERS[extension](uploaded_file)
    else:
        extension = "url"
        raw_text = DATA_LOADERS[extension](url)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Text")
        st.text_area(
            f"Number of characters before cleaning: {len(raw_text)}",
            f"{raw_text[:500]} . . .",
        )

    clean_text = DATA_CLEANERS[extension](raw_text)
    with col2:
        st.subheader("Cleaned Text")
        st.text_area(
            f"Number of characters after cleaning: {len(clean_text)}",
            f"{clean_text[:500]} . . .",
        )
    st.session_state["clean_text"] = clean_text

if "clean_text" in st.session_state:
    clean_text = st.session_state["clean_text"]

    st.divider()
    st.header("Text Optimization and Speech Settings")
    st.divider()

    # Text optimization settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Optimization")
        should_optimize_text = st.checkbox("Optimize text for speech", value=True)
        if should_optimize_text:
            optimization_model = st.selectbox(
                "Text Optimization Model",
                options=[m.value for m in TextOptimizationModel],
                format_func=lambda x: "Local LLM" if x == "local_llm" else "OpenRouter (Cloud)"
            )
            
            if optimization_model == TextOptimizationModel.OPENROUTER.value:
                openrouter_api_key = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    help="Get your API key from https://openrouter.ai/keys"
                )
                
                if openrouter_api_key:
                    try:
                        models = asyncio.run(get_available_models(openrouter_api_key))
                        model_options = [
                            f"{model['name']} (free)" 
                            if not model['pricing'] or (not model['pricing'].get('prompt') and not model['pricing'].get('completion'))
                            else f"{model['name']} (${(float(model['pricing'].get('prompt', 0)) + float(model['pricing'].get('completion', 0)))*1000000:.2f}/M tokens)" 
                            for model in models
                        ]
                        selected_model = st.selectbox(
                            "Select OpenRouter Model",
                            model_options,
                            help="Choose a model for text optimization. Prices shown are per 1 million tokens."
                        )
                        # Get the selected model ID
                        selected_model_idx = model_options.index(selected_model)
                        selected_model_id = models[selected_model_idx]["id"]
                    except Exception as e:
                        st.error(f"Error fetching models: {str(e)}")
                        selected_model_id = None
                else:
                    st.warning("Please enter your OpenRouter API key to see available models.")
                    selected_model_id = None
            
            optimization_level = st.select_slider(
                "Optimization Level",
                options=["basic", "moderate", "aggressive"],
                value="moderate"
            )

    with col2:
        st.subheader("Speech Settings")
        voice_profile = st.selectbox(
            "Voice Profile",
            options=["af_sarah", "af_bella", "af_grace", "am_michael", "am_james", "am_john",
                    "bf_sarah", "bf_bella", "bf_grace", "bm_michael", "bm_james", "bm_john"],
            index=0,
            help="a = American English, b = British English; f = female, m = male"
        )
        
        st.write("Speech Parameters:")
        speed = st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        volume = st.slider("Volume", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    if st.button("Generate Speech", on_click=gen_button_clicked):
        # Create placeholders for progress tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        
        with st.spinner("Processing..."):
            # Text optimization if enabled
            if should_optimize_text:
                status_placeholder.info("Optimizing text...")
                progress_bar = st.progress(0)
                
                # Create temporary files for input
                temp_input = Path("temp_input.txt")
                temp_input.write_text(clean_text)
                
                # Create config object for text optimization
                config = Config(
                    input_file=str(temp_input),  # Use actual file path
                    output_file="temp_output.wav",  # Placeholder, not used for streaming
                    optimize_text=True,
                    text_optimization_model=TextOptimizationModel(optimization_model),
                    optimization_level=optimization_level,
                    text_model="bartowski/Qwen2.5-7B-Instruct-GGUF",  # Model path for local LLM
                    openrouter_api_key=openrouter_api_key if optimization_model == TextOptimizationModel.OPENROUTER.value else None,
                    openrouter_model=selected_model_id if optimization_model == TextOptimizationModel.OPENROUTER.value else None
                )
                
                try:
                    # Split text into chunks to show progress
                    chunks = clean_text.split("\n\n")
                    total_chunks = len(chunks)
                    processed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        # Update progress
                        progress = (i + 1) / total_chunks
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Optimizing chunk {i + 1}/{total_chunks}")
                        
                        if i == total_chunks - 1:
                            status_placeholder.success("Text optimization completed!")
                    
                    # Run the async function in the event loop
                    processed_text = asyncio.run(optimize_text(clean_text, config))
                except Exception as e:
                    st.error(f"Error during text optimization: {str(e)}")
                    processed_text = clean_text  # Fallback to original text
                finally:
                    # Clean up temporary file
                    temp_input.unlink(missing_ok=True)
            else:
                processed_text = clean_text

            st.session_state.processed_text = processed_text
            
            # Show processed text
            text_placeholder.subheader("Processed Text")
            text_placeholder.text_area("Final text for speech synthesis:", processed_text)

            # Text to speech conversion
            status_placeholder.info("Loading speech model...")
            speech_model = load_text_to_speech_model(voice_profile=voice_profile)
            
            # Create progress bar for speech generation
            speech_progress = st.progress(0)
            status_placeholder.info("Generating speech...")
            
            # Split text into smaller segments for progress tracking
            segments = processed_text.split(". ")
            total_segments = len(segments)
            audio_segments = []
            
            speech_params = SpeechParams(speed=speed, volume=volume)
            
            for i, segment in enumerate(segments):
                if segment.strip():  # Only process non-empty segments
                    # Update progress
                    progress = (i + 1) / total_segments
                    speech_progress.progress(progress)
                    status_placeholder.info(f"Generating speech segment {i + 1}/{total_segments}")
                    
                    # Generate speech for segment
                    segment_audio = text_to_speech(
                        segment + ".",  # Add back the period
                        speech_model,
                        voice_profile,
                        speech_params
                    )
                    audio_segments.append(segment_audio)
            
            # Combine all audio segments
            status_placeholder.info("Finalizing audio...")
            audio = np.concatenate(audio_segments) if audio_segments else np.array([])
            status_placeholder.success("Speech generation completed!")
            
            st.session_state.audio = audio
            
            # Play the audio
            st.subheader("Generated Speech")
            st.audio(audio, sample_rate=sample_rate)

    if st.session_state.generate_button and st.session_state.audio is not None:
        # Download buttons
        audio_wav = numpy_to_wav(st.session_state.audio, sample_rate)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.download_button(
                label="Save Audio File",
                data=audio_wav,
                file_name="speech.wav",
            ):
                st.success("Audio saved!")
        
        with col2:
            if st.download_button(
                label="Save Processed Text",
                data=st.session_state.processed_text,
                file_name="processed_text.txt",
            ):
                st.success("Text saved!")
