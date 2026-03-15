# main.py  —  Alixir | Phase 2: Pydantic Validation + Dummy Router
#
# PURPOSE:
#   This runner no longer calls the video tool directly.
#   Instead, it simulates what the future LLM will do in Phase 3:
#
#       dummy JSON plan -> Pydantic validation -> router -> tool execution
#
#   This lets us prove the data-contract and routing layers work before we
#   connect the local Ollama model.

import os
import sys

from pydantic import ValidationError

from core.router import execute_edit_plan
from core.schemas import EditDecisionList


# ════════════════════════════════════════════════════════════════════════════
#  HARDCODED CONFIGURATION  (Phase 2 dummy payload — replaced by Ollama later)
# ════════════════════════════════════════════════════════════════════════════

# Path to the source video you want to cut.
# Place your test .mp4 in data/input/ and update this filename.
INPUT_VIDEO = os.path.join("data", "input", "sample.mp4")

# Where the trimmed output should be saved.
OUTPUT_VIDEO = os.path.join("data", "output", "phase2_cut.mp4")

# The time window we want to keep, in seconds.
# Example: cut from the 5-second mark to the 15-second mark.
START_TIME = 5.0   # seconds
END_TIME   = 15.0  # seconds


# This dummy payload mirrors the shape that Ollama will produce in Phase 3.
DUMMY_EDIT_PLAN = {
    "phase": "phase_2",
    "user_request": "Cut the input video from 5 seconds to 15 seconds.",
    "actions": [
        {
            "tool_name": "cut_video_segment",
            "input_path": INPUT_VIDEO,
            "output_path": OUTPUT_VIDEO,
            "parameters": {
                "start_time": START_TIME,
                "end_time": END_TIME,
            },
        }
    ],
}


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Phase 2 runner: validates a dummy plan and routes it to the correct tool.
    Returns exit code 0 on success, 1 on any handled error.
    """

    print("=" * 60)
    print("  Alixir — Phase 2: Pydantic Validation + Dummy Router")
    print("=" * 60)
    print(f"  Input  : {INPUT_VIDEO}")
    print(f"  Output : {OUTPUT_VIDEO}")
    print(f"  Range  : {START_TIME}s  →  {END_TIME}s")
    print("=" * 60)

    try:
        # ── Step 1: Validate the dummy JSON against the strict Pydantic schema ──
        validated_plan = EditDecisionList.model_validate(DUMMY_EDIT_PLAN)

        print(f"[main] Validated {len(validated_plan.actions)} action(s).")

        # ── Step 2: Hand the validated plan to the router for execution ──
        result_paths = execute_edit_plan(validated_plan)

        print()
        print(f"[main] Phase 2 complete. Output file: {result_paths[-1]}")
        return 0

    except ValidationError as e:
        # The dummy payload shape did not match our strict schema.
        print("\n[main] ERROR — Payload failed Pydantic validation.")
        print(e)
        return 1

    except FileNotFoundError as e:
        # The tool layer still validates real file existence at runtime.
        print(f"\n[main] ERROR — {e}")
        print(
            "[main] TIP: Place a test .mp4 at 'data/input/sample.mp4' "
            "and re-run."
        )
        return 1

    except ValueError as e:
        # Either the router saw an unregistered tool or the tool rejected values.
        print(f"\n[main] ERROR — {e}")
        return 1


# ── Standard Python idiom: only run main() when this file is executed directly,
#    not when it is imported by another module (e.g., a test). ──
if __name__ == "__main__":
    sys.exit(main())
