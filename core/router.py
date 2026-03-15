# core/router.py
#
# PURPOSE:
#   This file is the execution bridge between validated Pydantic objects and
#   the actual Python video-editing tools.
#
#   The router does not parse raw JSON. It assumes validation already happened
#   in core/schemas.py. Its only job is:
#       receive AgentAction -> find matching tool -> call tool safely
#
# PARTNER NOTE:
#   When new tools are added, register them in TOOL_REGISTRY and extend the
#   AgentAction/tool parameter schema in core/schemas.py.

from core.schemas import AgentAction, EditDecisionList
from tools.video_cutter import cut_video_segment


# ── Central registry of tool names to Python callables ──
# This keeps the architecture modular. Your partner can add a new tool file,
# import the function here, and register it under a string key.
TOOL_REGISTRY = {
    "cut_video_segment": cut_video_segment,
}


def execute_action(action: AgentAction) -> str:
    """
    Execute one validated AgentAction and return the tool's output path.
    """

    if action.tool_name not in TOOL_REGISTRY:
        raise ValueError(f"[router] Unregistered tool: {action.tool_name}")

    tool_function = TOOL_REGISTRY[action.tool_name]

    print(f"[router] Executing tool: {action.tool_name}")
    print(f"[router] Input file   : {action.input_path}")
    print(f"[router] Output file  : {action.output_path}")

    return tool_function(
        input_path=action.input_path,
        output_path=action.output_path,
        start_time=action.parameters.start_time,
        end_time=action.parameters.end_time,
    )


def execute_edit_plan(plan: EditDecisionList) -> list[str]:
    """
    Execute each action in order and collect each resulting output path.
    """

    results = []

    for index, action in enumerate(plan.actions, start=1):
        print(f"[router] Action {index}/{len(plan.actions)}")
        result_path = execute_action(action)
        results.append(result_path)

    return results
