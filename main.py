# main.py  —  Alixir | Phase 1: The "Dumb" Cut
#
# PURPOSE:
#   This is the Phase 1 runner. Its only job is to prove that our video
#   cutting tool works end-to-end before we introduce any AI or routing.
#
#   Think of this like a "hello world" for the pipeline:
#       Input video  →  hardcoded timestamps  →  cut clip  →  output file
#
#   In Phase 4, this file will be replaced by the full agentic loop where
#   the timestamps come from the LLM rather than being typed by hand.

import os
import sys

# ── Import our first tool from the /tools directory ──
# Because tools/__init__.py exists, Python treats /tools as a package,
# letting us use clean dot-notation imports.
from tools.video_cutter import cut_video_segment


# ════════════════════════════════════════════════════════════════════════════
#  HARDCODED CONFIGURATION  (Phase 1 only — replaced by LLM output in Phase 3)
# ════════════════════════════════════════════════════════════════════════════

# Path to the source video you want to cut.
# Place your test .mp4 in data/input/ and update this filename.
INPUT_VIDEO = os.path.join("data", "input", "sample.mp4")

# Where the trimmed output should be saved.
OUTPUT_VIDEO = os.path.join("data", "output", "phase1_cut.mp4")

# The time window we want to keep, in seconds.
# Example: cut from the 5-second mark to the 15-second mark.
START_TIME = 5.0   # seconds
END_TIME   = 15.0  # seconds


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Phase 1 runner: calls cut_video_segment with hardcoded values.
    Returns exit code 0 on success, 1 on any handled error.
    """

    print("=" * 60)
    print("  Alixir — Phase 1: Hardcoded Video Cut")
    print("=" * 60)
    print(f"  Input  : {INPUT_VIDEO}")
    print(f"  Output : {OUTPUT_VIDEO}")
    print(f"  Range  : {START_TIME}s  →  {END_TIME}s")
    print("=" * 60)

    try:
        # ── Call the tool. In later phases this call will originate from
        #    the router, which will receive its arguments from Pydantic. ──
        result_path = cut_video_segment(
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
            start_time=START_TIME,
            end_time=END_TIME,
        )

        print()
        print(f"[main] Phase 1 complete. Output file: {result_path}")
        return 0

    except FileNotFoundError as e:
        # User forgot to put a video in data/input/ — give a helpful message.
        print(f"\n[main] ERROR — {e}")
        print(
            "[main] TIP: Place a test .mp4 at 'data/input/sample.mp4' "
            "and re-run."
        )
        return 1

    except ValueError as e:
        # Timestamp logic error — either our hardcoded values are wrong,
        # or the video is shorter than expected.
        print(f"\n[main] ERROR — {e}")
        return 1


# ── Standard Python idiom: only run main() when this file is executed directly,
#    not when it is imported by another module (e.g., a test). ──
if __name__ == "__main__":
    sys.exit(main())
