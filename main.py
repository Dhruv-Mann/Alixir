# main.py  —  Alixir | Phase 3: Local Ollama Integration
#
# PURPOSE:
#   This runner asks the local Ollama model to generate a JSON edit plan,
#   validates that plan with Pydantic, and then executes it through the router.
#
#   The data path is now:
#       system prompt + user prompt -> Ollama -> validated JSON plan -> router

import os
import sys

from pydantic import ValidationError

from core.ollama_client import generate_edit_plan
from core.router import execute_edit_plan


# ════════════════════════════════════════════════════════════════════════════
#  HARDCODED CONFIGURATION  (Phase 3 test input for the local model)
# ════════════════════════════════════════════════════════════════════════════

# Path to the source video you want to cut.
# Place your test .mp4 in data/input/ and update this filename.
INPUT_VIDEO = os.path.join("data", "input", "sample.mp4")

# Where the trimmed output should be saved.
OUTPUT_VIDEO = os.path.join("data", "output", "phase3_cut.mp4")

# The local model already installed in Ollama.
OLLAMA_MODEL = "qwen2.5-coder:7b"

# The system prompt file that defines the planner's behavioral rules.
SYSTEM_PROMPT_PATH = os.path.join("prompts", "edit_planner_system_prompt.txt")

# A simple natural-language request for testing the full model path.
USER_REQUEST = "Cut the input video from 5 seconds to 15 seconds."


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Phase 3 runner: asks Ollama for a plan, validates it, then routes it.
    Returns exit code 0 on success, 1 on any handled error.
    """

    print("=" * 60)
    print("  Alixir — Phase 3: Local Ollama Integration")
    print("=" * 60)
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Input  : {INPUT_VIDEO}")
    print(f"  Output : {OUTPUT_VIDEO}")
    print(f"  Request: {USER_REQUEST}")
    print("=" * 60)

    try:
        # ── Step 1: Ask the local model for a JSON plan and validate it ──
        validated_plan = generate_edit_plan(
            model_name=OLLAMA_MODEL,
            system_prompt_path=SYSTEM_PROMPT_PATH,
            user_request=USER_REQUEST,
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
        )

        print(f"[main] Validated {len(validated_plan.actions)} action(s).")

        # ── Step 2: Hand the validated plan to the router for execution ──
        result_paths = execute_edit_plan(validated_plan)

        print()
        print(f"[main] Phase 3 complete. Output file: {result_paths[-1]}")
        return 0

    except ValidationError as e:
        # Ollama responded, but its JSON did not satisfy the schema contract.
        print("\n[main] ERROR — Ollama output failed Pydantic validation.")
        print(e)
        return 1

    except RuntimeError as e:
        # Covers connection failures, empty responses, or invalid JSON text.
        print(f"\n[main] ERROR — {e}")
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
