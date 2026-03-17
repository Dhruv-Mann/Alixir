# main.py  —  Alixir | Phase 4: Transcription-Aware Plan Generation
#
# PURPOSE:
#   This runner extracts and transcribes audio from a video, then asks the local Ollama
#   model to generate a JSON edit plan informed by the transcript, validates that plan
#   with Pydantic, and executes it through the router.
#
#   The data path is now:
#       video -> transcriber -> audio transcript
#       transcript + system prompt + user prompt -> Ollama -> validated JSON plan -> router

import os
import sys

from pydantic import ValidationError

from core.transcriber import transcribe_audio_from_video
from core.ollama_client import generate_edit_plan
from core.router import execute_edit_plan


# ════════════════════════════════════════════════════════════════════════════
#  HARDCODED CONFIGURATION  (Phase 4 test input)
# ════════════════════════════════════════════════════════════════════════════

# Path to the source video you want to cut.
# Place your test .mp4 in data/input/ and update this filename.
INPUT_VIDEO = os.path.join("data", "input", "sample.mp4")

# Where the trimmed output should be saved.
OUTPUT_VIDEO = os.path.join("data", "output", "phase4_cut.mp4")

# The local model already installed in Ollama.
OLLAMA_MODEL = "qwen2.5-coder:7b"

# The system prompt file that defines the planner's behavioral rules (Phase 4 version).
SYSTEM_PROMPT_PATH = os.path.join("prompts", "edit_planner_system_prompt_phase4.txt")

# A simple natural-language request for testing the full model path.
USER_REQUEST = "Cut the input video from 5 seconds to 15 seconds."


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Phase 4 runner: transcribe audio, ask Ollama for a plan, validate, then route.
    Returns exit code 0 on success, 1 on any handled error.
    """

    print("=" * 60)
    print("  Alixir — Phase 4: Transcription-Aware Plan Generation")
    print("=" * 60)
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Input  : {INPUT_VIDEO}")
    print(f"  Output : {OUTPUT_VIDEO}")
    print(f"  Request: {USER_REQUEST}")
    print("=" * 60)

    try:
        # ── Step 1: Extract and transcribe audio from the input video ──
        transcript = transcribe_audio_from_video(INPUT_VIDEO)
        print(f"[main] Transcript extracted: {len(transcript)} characters.")

        # ── Step 2: Ask the local model for a JSON plan using the transcript ──
        validated_plan = generate_edit_plan(
            model_name=OLLAMA_MODEL,
            system_prompt_path=SYSTEM_PROMPT_PATH,
            user_request=USER_REQUEST,
            transcript=transcript,
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
        )

        print(f"[main] Validated {len(validated_plan.actions)} action(s).")

        # ── Step 3: Hand the validated plan to the router for execution ──
        result_paths = execute_edit_plan(validated_plan)

        print()
        print(f"[main] Phase 4 complete. Output file: {result_paths[-1]}")
        return 0

    except FileNotFoundError as e:
        print(f"\n[main] ERROR — {e}")
        print(f"[main] TIP: Place a test .mp4 at {INPUT_VIDEO} and re-run.")
        return 1

    except RuntimeError as e:
        # GPU or transcription failure
        print(f"\n[main] ERROR — Runtime error: {e}")
        return 1

    except ValidationError as e:
        # Ollama responded, but its JSON did not satisfy the schema contract.
        print("\n[main] ERROR — Ollama output failed Pydantic validation.")
        print(e)
        return 1

    except ValueError as e:
        # Either the router saw an unregistered tool or the tool rejected values.
        print(f"\n[main] ERROR — {e}")
        return 1


# ── Standard Python idiom: only run main() when this file is executed directly,
#    not when it is imported by another module (e.g., a test). ──
if __name__ == "__main__":
    sys.exit(main())
