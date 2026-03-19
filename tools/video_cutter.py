# tools/video_cutter.py
#
# PURPOSE:
#   This is the first "Tool" in Alixir — a self-contained function that takes
#   a video file and cuts a segment out of it using hardcoded (or passed-in)
#   timestamps.
#
#   This function is CALLED BY THE ROUTER, which receives its arguments from
#   the LLM via a validated Pydantic object.
#   The implementation remains standalone and deterministic for easy testing.
#
# PARTNER NOTE:
#   Every tool your partner writes should follow this same signature pattern:
#       def tool_name(input_path: str, output_path: str, **kwargs) -> str
#   This makes it trivial to register new tools in the router later.

import os
from moviepy import VideoFileClip


def cut_video_segment(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    Cuts a single segment from a video file between start_time and end_time.

    Args:
        input_path  : Absolute or relative path to the source .mp4 file.
        output_path : Where to save the trimmed clip.
        start_time  : Start of the cut in seconds (e.g. 10.5 = 10 seconds 500 ms).
        end_time    : End of the cut in seconds.

    Returns:
        The output_path string on success, so the caller can chain operations.

    Raises:
        FileNotFoundError : If input_path does not exist.
        ValueError        : If timestamps are logically invalid.
    """

    # ── Guard: make sure the source file actually exists before even trying ──
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"[video_cutter] Source file not found: '{input_path}'"
        )

    # ── Guard: timestamps must make sense (start before end, both non-negative) ──
    if start_time < 0 or end_time < 0:
        raise ValueError(
            f"[video_cutter] Timestamps cannot be negative. "
            f"Got start={start_time}, end={end_time}"
        )
    if start_time >= end_time:
        raise ValueError(
            f"[video_cutter] start_time must be less than end_time. "
            f"Got start={start_time}, end={end_time}"
        )

    # ── Ensure the output directory exists; create it if it doesn't ──
    # os.makedirs with exist_ok=True is safe to call even if the folder already exists.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[video_cutter] Loading clip: {input_path}")
    print(f"[video_cutter] Cutting from {start_time}s  →  {end_time}s")

    # ── Load the full video into MoviePy ──
    # VideoFileClip reads the video file and prepares it for manipulation.
    # It does NOT load the entire video into RAM — it streams it. This is
    # important for large files on a memory-constrained machine.
    with VideoFileClip(input_path) as clip:

        # ── Validate that our timestamps don't exceed the video duration ──
        if end_time > clip.duration:
            raise ValueError(
                f"[video_cutter] end_time ({end_time}s) exceeds video "
                f"duration ({clip.duration:.2f}s)."
            )

        # ── subclipped() returns a NEW clip object spanning [start, end] ──
        # Nothing is written to disk yet — this is still in-memory.
        trimmed = clip.subclipped(start_time, end_time)

        # ── write_videofile() is where the actual rendering happens ──
        # codec="libx264" : standard H.264 encoding — widely compatible.
        # audio_codec="aac": standard AAC audio — pairs with H.264.
        # logger=None      : silences MoviePy's verbose ffmpeg progress bar
        #                    so our own print statements stay readable.
        trimmed.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

    print(f"[video_cutter] Done. Saved to: {output_path}")

    # Return the output path so the caller (and later the router) can confirm
    # where the result file landed.
    return output_path
