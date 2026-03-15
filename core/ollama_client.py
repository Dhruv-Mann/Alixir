# core/ollama_client.py
#
# PURPOSE:
#   This file is the Phase 3 bridge between the local Ollama server and the
#   strict Pydantic tool-planning contract used by Alixir.
#
#   The key workflow is:
#       load system prompt -> build user prompt -> call Ollama once
#       -> parse model text into JSON -> validate with Pydantic
#
# MEMORY NOTE:
#   We keep the model warm briefly to avoid repeated cold-load CUDA failures on
#   newer GPUs. Heavy video rendering still runs after this lightweight planning
#   step, and keep-alive duration is configurable.

import json
import os
from pathlib import Path

from ollama import Client, ResponseError
from pydantic import ValidationError

from core.schemas import EditDecisionList


def request_chat_response(
    client: Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
):
    """Make one Ollama chat request and optionally retry on CPU if allowed."""

    allow_cpu_fallback = os.getenv("ALIXIR_ALLOW_CPU_FALLBACK", "0") == "1"
    num_ctx = int(os.getenv("ALIXIR_OLLAMA_NUM_CTX", "4096"))
    num_batch = int(os.getenv("ALIXIR_OLLAMA_NUM_BATCH", "128"))
    keep_alive = os.getenv("ALIXIR_OLLAMA_KEEP_ALIVE", "10m")

    try:
        return client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=EditDecisionList.model_json_schema(),
            options={
                "temperature": 0,
                "num_ctx": num_ctx,
                "num_batch": num_batch,
            },
            keep_alive=keep_alive,
        )
    except ResponseError as exc:
        if "CUDA error" not in str(exc):
            raise

        if not allow_cpu_fallback:
            raise RuntimeError(
                "[ollama_client] Ollama hit a CUDA runtime error while loading "
                "the model on GPU. GPU execution is required by default. "
                "If you want temporary CPU fallback for debugging, set "
                "ALIXIR_ALLOW_CPU_FALLBACK=1."
            ) from exc

        print("[ollama_client] GPU inference failed. Retrying on CPU.")
        return client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=EditDecisionList.model_json_schema(),
            options={
                "temperature": 0,
                "num_ctx": num_ctx,
                "num_batch": num_batch,
                "num_gpu": 0,
            },
            keep_alive=keep_alive,
        )


def load_system_prompt(prompt_path: str) -> str:
    """Read the system prompt file from disk."""
    return Path(prompt_path).read_text(encoding="utf-8").strip()


def build_user_prompt(
    user_request: str,
    input_path: str,
    output_path: str,
) -> str:
    """
    Build the concrete planning prompt sent to the model.

    Keep the prompt short and concrete so the local model spends less context
    budget on instructions and more on producing the edit decision.
    """

    return (
        "Plan one local video edit request using the available tool.\n\n"
        f"User request: {user_request}\n"
        f"Input path: {input_path}\n"
        f"Output path: {output_path}\n\n"
        "Available tool:\n"
        "- cut_video_segment(input_path, output_path, start_time, end_time)\n\n"
        "Output requirements:\n"
        "- Set phase to \"phase_3\".\n"
        "- Use tool_name \"cut_video_segment\".\n"
        "- Use the exact input_path and output_path above.\n"
        "- Put cut parameters inside the nested parameters object.\n"
        "- Use numeric seconds for start_time and end_time.\n"
        "- Choose the most direct single cut that satisfies the request.\n"
    )


def extract_json_text(raw_response: str) -> str:
    """
    Extract the JSON payload even if the model wraps it in code fences.
    """

    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        raise RuntimeError("[ollama_client] Model response did not contain JSON.")

    return cleaned[first_brace : last_brace + 1]


def generate_edit_plan(
    model_name: str,
    system_prompt_path: str,
    user_request: str,
    input_path: str,
    output_path: str,
    ollama_host: str = "http://127.0.0.1:11434",
) -> EditDecisionList:
    """
    Ask the local Ollama model for a JSON edit plan and validate it.
    """

    system_prompt = load_system_prompt(system_prompt_path)
    user_prompt = build_user_prompt(
        user_request=user_request,
        input_path=input_path,
        output_path=output_path,
    )

    client = Client(host=ollama_host)

    try:
        response = request_chat_response(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "[ollama_client] Failed to contact the local Ollama server. "
            "Make sure the Ollama app is running, the model is installed, "
            "and the local runtime can load the model."
        ) from exc

    raw_content = response["message"]["content"]

    if not raw_content or not raw_content.strip():
        raise RuntimeError("[ollama_client] Ollama returned an empty response.")

    json_text = extract_json_text(raw_content)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "[ollama_client] Model response was not valid JSON after extraction."
        ) from exc

    try:
        return EditDecisionList.model_validate(payload)
    except ValidationError:
        # Re-raise the exact Pydantic validation error so main.py can print it.
        raise
