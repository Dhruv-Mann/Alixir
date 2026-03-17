# core/transcriber.py
#
# PURPOSE:
#   Extract audio from a video file and transcribe it using faster-whisper.
#   This is the preprocessing step for Phase 4: transcription-aware plan generation.
#
#   Design principle: Keep transcripts lean to protect GPU stability during Ollama planning.
#   Long transcripts are truncated to preserve model context for planning decisions.

import os
from pathlib import Path

from faster_whisper import WhisperModel


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Whisper model size. Options: tiny, base, small, medium, large
# For RTX 5060 with 8GB VRAM, "base" is the safest choice; "small" is acceptable.
WHISPER_MODEL_SIZE = os.getenv("ALIXIR_WHISPER_MODEL_SIZE", "base")

# Maximum length of transcript to pass to Ollama planner (in characters).
# This protects GPU memory during planning by preventing huge prompts.
# Adjust based on your num_ctx setting (currently 4096 tokens ≈ 16384 chars).
MAX_TRANSCRIPT_LENGTH = int(os.getenv("ALIXIR_MAX_TRANSCRIPT_LENGTH", "2000"))

# Whether to use GPU for transcription. If False, uses CPU.
TRANSCRIBE_ON_GPU = os.getenv("ALIXIR_TRANSCRIBE_ON_GPU", "1") == "1"


# ════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def transcribe_audio_from_video(video_path: str) -> str:
    """
    Extract audio from a video file and transcribe using faster-whisper.

    Args:
        video_path: Path to the input video file.

    Returns:
        Transcribed text (truncated if longer than MAX_TRANSCRIPT_LENGTH).

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If transcription fails (e.g., no audio stream, model load failure).
    """

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[transcriber] Loading Whisper model: {WHISPER_MODEL_SIZE}")
    try:
        device = "cuda" if TRANSCRIBE_ON_GPU else "cpu"
        model = WhisperModel(
            model_size_or_path=WHISPER_MODEL_SIZE,
            device=device,
            compute_type="float16" if TRANSCRIBE_ON_GPU else "int8",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")

    print(f"[transcriber] Transcribing audio from: {video_path}")
    try:
        segments, info = model.transcribe(video_path, language="en")
        # Collect all segment text into one transcript
        full_transcript = " ".join([segment.text for segment in segments])
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

    # Truncate if too long to protect GPU memory during planning
    if len(full_transcript) > MAX_TRANSCRIPT_LENGTH:
        print(
            f"[transcriber] Transcript truncated from {len(full_transcript)} "
            f"to {MAX_TRANSCRIPT_LENGTH} characters."
        )
        full_transcript = full_transcript[:MAX_TRANSCRIPT_LENGTH]

    print(f"[transcriber] Transcription complete. Length: {len(full_transcript)} chars.")
    return full_transcript
