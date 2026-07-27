"""
entity_extraction.py

Reusable helper to extract structured entities from free text using a local
llama.cpp server (OpenAI-compatible /v1/chat/completions endpoint).

Env vars required:
    LLAMA_CPP_LLM_MODEL   e.g. "qwen3.5-uncensored:2b"
    LLAMA_CPP_LLM_HOST    e.g. "http://192.168.1.50:8080"
"""

import json
import os
from typing import Type, TypeVar

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.logger import logger
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Client setup — reads env vars once at import time
# ---------------------------------------------------------------------------
_LLAMA_CPP_LLM_MODEL = os.environ["LLAMA_CPP_LLM_MODEL"]

_client = get_llm_client()

T = TypeVar("T", bound=BaseModel)


def _to_strict_json_schema(model: Type[BaseModel]) -> dict:
    """
    Convert a Pydantic model's schema into "strict" JSON-schema form, which is
    what the OpenAI-style response_format expects for reliable constrained
    decoding:
      - every object gets additionalProperties: false
      - every property is marked required (strict mode requires this)
    Works recursively through nested models, lists, and $defs.
    """
    schema = model.model_json_schema()

    def _make_strict(node: dict) -> None:
        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            for prop_schema in node["properties"].values():
                _make_strict(prop_schema)
        if "items" in node:
            _make_strict(node["items"])
        for defs_key in ("$defs", "definitions"):
            if defs_key in node:
                for def_schema in node[defs_key].values():
                    _make_strict(def_schema)

    _make_strict(schema)
    return schema


def extract_entities_from_text(
    text: str,
    model: Type[T],
    *,
    temperature: float = 0.0,
    extra_instructions: str = "",
) -> T:
    """
    Extract structured entities from `text`, constrained to the shape of `model`.

    Args:
        text: the input text to extract entities from.
        model: a Pydantic model class describing the fields/labels to extract.
        temperature: sampling temperature (0.0 = deterministic, recommended for extraction).
        extra_instructions: optional extra guidance appended to the system prompt
            (e.g. "Dates must be ISO 8601", "Only extract explicitly named people").

    Returns:
        An instance of `model` populated from the LLM's output.

    Raises:
        ValueError: if the server response is missing, not valid JSON, or doesn't
            match the schema of `model`.
    """
    schema = _to_strict_json_schema(model)

    system_prompt = (
        "You are an information extraction engine. "
        "Extract entities from the user's text strictly following the given JSON schema. "
        "If a field is not present in the text, use a sensible empty/default value. "
        "Respond with JSON only — no commentary, no markdown fences."
    )
    if extra_instructions:
        system_prompt += f"\n\nAdditional instructions: {extra_instructions}"

    logger.info(
        "Extracting '%s' entities | model=%s | text_len=%d",
        model.__name__,
        _LLAMA_CPP_LLM_MODEL,
        len(text),
    )
    logger.debug("Input text: %s", text)

    stream: Stream[ChatCompletionChunk] = _client.chat.completions.create(
        model=_LLAMA_CPP_LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        stream=True,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )

    # Extract the response content
    content = ""
    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                content += delta.reasoning_content
                logger.orange(delta.reasoning_content, flush=True, end="")

            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                content += delta.content
                logger.teal(delta.content, flush=True, end="")

    try:
        parsed = model.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(
            "Failed to parse/validate output against %s: %s", model.__name__, e
        )
        raise ValueError(
            f"LLM output did not match schema '{model.__name__}': {e}"
        ) from e

    logger.info("Successfully extracted %s", model.__name__)
    return parsed


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from typing import List

    class PersonEntities(BaseModel):
        names: List[str]
        organizations: List[str]
        locations: List[str]

    sample_text = "Alice Johnson met with Bob from Acme Corp in San Francisco."
    result = extract_entities_from_text(sample_text, PersonEntities)
    print(result.model_dump_json(indent=2))
