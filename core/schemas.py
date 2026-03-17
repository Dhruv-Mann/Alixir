# core/schemas.py
#
# PURPOSE:
#   This file defines the strict JSON contract between the future LLM output
#   and the Python tool layer. In Alixir, the model does not get to call tools
#   directly. It must first produce JSON that matches these Pydantic models.
#
#   That gives us a safety gate:
#       raw JSON -> Pydantic validation -> safe Python objects -> router -> tool
#
# PHASE EVOLUTION:
#   Phase 3: Local LLM produces JSON against schema. Supports cut_video_segment.
#   Phase 4: Adds transcription layer. Expanded to support phase_3 and phase_4 outputs.
#   In later phases, new tool names and parameter models can be added here.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CutVideoParameters(BaseModel):
    """
    Parameters required by the cut_video_segment tool.

    Keeping tool-specific parameters in their own model makes the schema easier
    to extend later when more tools are added.
    """

    model_config = ConfigDict(extra="forbid")

    start_time: float = Field(
        ...,
        ge=0,
        description="Start time of the kept clip, in seconds.",
    )
    end_time: float = Field(
        ...,
        gt=0,
        description="End time of the kept clip, in seconds.",
    )

    @model_validator(mode="after")
    def validate_time_range(self):
        """Ensure the requested cut window is logically valid."""
        if self.start_time >= self.end_time:
            raise ValueError(
                "start_time must be smaller than end_time for a cut action."
            )
        return self


class AgentAction(BaseModel):
    """
    A single action the agent wants the router to execute.

    Later, the LLM will emit one or more of these objects in JSON form.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_name: Literal["cut_video_segment"] = Field(
        ...,
        description="Registered Python tool to execute.",
    )
    input_path: str = Field(
        ...,
        min_length=1,
        description="Source video path to feed into the tool.",
    )
    output_path: str = Field(
        ...,
        min_length=1,
        description="Destination path for the rendered result.",
    )
    parameters: CutVideoParameters


class EditDecisionList(BaseModel):
    """
    The full validated plan emitted by the agent for one editing request.

    In Phase 3 and later, the local Ollama model must emit JSON that matches this exact
    structure before the router is allowed to execute anything.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    phase: Literal["phase_3", "phase_4"] = Field(
        ...,
        description="Marks which development phase produced this payload.",
    )
    user_request: str = Field(
        ...,
        min_length=1,
        description="Original human instruction that led to these actions.",
    )
    actions: list[AgentAction] = Field(
        ...,
        min_length=1,
        description="Ordered list of actions the router should execute.",
    )
