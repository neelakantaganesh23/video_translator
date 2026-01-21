import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel
from moviepy import VideoFileClip


@st.cache_resource
def load_whisper_model(model_size: str = "small"):
    """
    Load the Whisper model once and cache it.

    Using 'small' is a good balance for CPU-only machines.
    You can change to 'base' for faster but slightly worse quality.
    """
    # device="auto" will use GPU if available, otherwise CPU
    return WhisperModel(model_size, device="auto", compute_type="int8")


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio track from the input video and save as WAV."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    clip.close()


def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Run Whisper transcription on an audio file and return plain text."""
    model = load_whisper_model()

    segments, _info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        best_of=5,
    )

    # Concatenate all segment texts into a single transcript
    return " ".join(segment.text.strip() for segment in segments)


def main():
    st.title("Video Translation (Free Stack Prototype)")
    st.write(
        "Upload a video, and this app will transcribe the audio using a local Whisper model. "
        "Later we will add translation and re-dubbing."
    )

    st.warning(
        "For this to work, you need FFmpeg installed on your system (required by MoviePy). "
        "On Windows, you can install it from `https://ffmpeg.org` and ensure `ffmpeg.exe` is on your PATH."
    )

    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "mkv", "mov", "avi"]
    )

    language = st.text_input(
        "Spoken language code in the video (ISO code, e.g. 'en', 'es', 'fr')",
        value="en",
    )

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        if st.button("Transcribe video"):
            with st.spinner("Processing video (this may take a while)..."):
                # Save uploaded file to a temporary location
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = os.path.join(tmpdir, uploaded_file.name)
                    audio_path = os.path.join(tmpdir, "audio.wav")

                    # Write video bytes to disk
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Extract audio
                    extract_audio_from_video(video_path, audio_path)

                    # Transcribe
                    try:
                        transcript = transcribe_audio(audio_path, language=language)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        return

                st.subheader("Transcript")
                st.text_area("Transcribed text", transcript, height=300)


if __name__ == "__main__":
    main()

