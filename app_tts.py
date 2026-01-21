import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
from moviepy import VideoFileClip


@st.cache_resource
def load_whisper_model(model_size: str = "small", compute_type: str = "int8"):
    return WhisperModel(model_size, device="auto", compute_type=compute_type)


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    clip.close()


@dataclass
class Segment:
    idx: int
    start: float
    end: float
    text: str


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        lines.append(str(seg.idx))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def whisper_to_english(
    audio_path: str,
    *,
    model_size: str,
    compute_type: str,
) -> tuple[str, list[Segment]]:
    model = load_whisper_model(model_size=model_size, compute_type=compute_type)
    raw_segments, _info = model.transcribe(
        audio_path,
        task="translate",  # Whisper built-in → English
        language=None,  # auto-detect
        beam_size=5,
        best_of=5,
        vad_filter=True,
    )

    segs: list[Segment] = []
    texts: list[str] = []
    for i, seg in enumerate(raw_segments, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue
        segs.append(Segment(idx=i, start=float(seg.start), end=float(seg.end), text=text))
        texts.append(text)

    return " ".join(texts), segs


def piper_tts_to_wav(*, text: str, model_path: Path, output_wav_path: Path) -> None:
    cmd = [
        "piper",
        "--model",
        str(model_path),
        "--output_file",
        str(output_wav_path),
    ]
    subprocess.run(
        cmd,
        input=text,
        text=True,
        check=True,
        capture_output=True,
    )


def main():
    st.title("Video → English Text + English Audio (Piper TTS)")
    st.write(
        "This version uses Whisper built-in translation to English and then generates English audio using Piper TTS."
    )

    st.warning(
        "Requires FFmpeg (MoviePy) and Piper TTS. Install deps with `pip install -r requirements.txt`."
    )

    uploaded_file = st.file_uploader(
        "Choose a video or audio file",
        type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a", "flac"],
        help="You can upload either a video (audio will be extracted) or an audio file directly.",
    )

    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox("Whisper model size", options=["base", "small", "medium"], index=1)
    with col2:
        compute_type = st.selectbox("Compute type", options=["int8", "float16", "float32"], index=0)

    piper_model_file = st.file_uploader("Piper English voice model (.onnx)", type=["onnx"])
    piper_config_file = st.file_uploader(
        "Piper model config (.onnx.json)",
        type=["json"],
        help="From the same voice download; required so Piper knows phonemes/sampling rate.",
    )

    if uploaded_file is None:
        return

    # Reset cached transcript if a different file is uploaded
    if st.session_state.get("last_video_name") != uploaded_file.name:
        st.session_state.pop("transcript_en", None)
        st.session_state.pop("segs", None)
        st.session_state["last_video_name"] = uploaded_file.name

    col_text, col_audio = st.columns(2)
    with col_text:
        if st.button("Generate English text", key="gen_text"):
            with st.spinner("Processing → English text..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Decide whether this is audio or video based on extension
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    tmp_input_path = os.path.join(tmpdir, uploaded_file.name)
                    audio_path = os.path.join(tmpdir, "audio.wav")

                    with open(tmp_input_path, "wb") as f:
                        f.write(uploaded_file.read())

                    if ext in [".mp3", ".wav", ".m4a", ".flac"]:
                        # Already audio; just point Whisper to it
                        audio_path = tmp_input_path
                    else:
                        # Treat as video; extract audio first
                        extract_audio_from_video(tmp_input_path, audio_path)

                    transcript_en, segs = whisper_to_english(
                        audio_path, model_size=model_size, compute_type=compute_type
                    )

                st.session_state["transcript_en"] = transcript_en
                st.session_state["segs"] = segs

    with col_audio:
        if st.button("Generate English audio (Piper)", key="gen_audio"):
            if "transcript_en" not in st.session_state or not st.session_state["transcript_en"]:
                st.warning("Run 'Generate English text' first.")
            elif piper_model_file is None or piper_config_file is None:
                st.info("Upload both the Piper `.onnx` model and matching `.onnx.json` config.")
            else:
                with st.spinner("Generating English audio with Piper..."):
                    try:
                        with tempfile.TemporaryDirectory() as tts_tmpdir:
                            tts_tmp = Path(tts_tmpdir)
                            model_path = tts_tmp / "voice.onnx"
                            config_path = tts_tmp / "voice.onnx.json"
                            out_wav = tts_tmp / "tts.wav"
                            model_path.write_bytes(piper_model_file.getvalue())
                            config_path.write_bytes(piper_config_file.getvalue())

                            piper_tts_to_wav(
                                text=st.session_state["transcript_en"],
                                model_path=model_path,
                                output_wav_path=out_wav,
                            )
                            wav_bytes = out_wav.read_bytes()
                    except subprocess.CalledProcessError as e:
                        st.error(
                            "Piper failed. Make sure `piper-tts` is installed and `piper` is on PATH. "
                            f"Details: {e.stderr or e}"
                        )
                        return
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
                        return

                st.success("English audio generated.")
                st.download_button(
                    "Download generated English audio (.wav)",
                    data=wav_bytes,
                    file_name="translated_en.wav",
                    mime="audio/wav",
                )

    # Show transcript/subtitles if available
    if "transcript_en" in st.session_state and st.session_state["transcript_en"]:
        transcript_en = st.session_state["transcript_en"]
        segs = st.session_state.get("segs", [])

        st.subheader("English text")
        st.text_area("Translated to English (Whisper)", transcript_en, height=220)
        st.download_button(
            "Download English transcript (.txt)",
            data=transcript_en.encode("utf-8"),
            file_name="translated_en.txt",
            mime="text/plain; charset=utf-8",
        )
        srt_text = segments_to_srt(segs)
        st.download_button(
            "Download English subtitles (.srt)",
            data=srt_text.encode("utf-8"),
            file_name="translated_en.srt",
            mime="application/x-subrip; charset=utf-8",
        )

        with st.expander("Show timestamped segments"):
            st.dataframe(
                [
                    {"#": s.idx, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                    for s in segs
                ],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

