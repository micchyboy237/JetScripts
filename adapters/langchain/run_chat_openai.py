"""Standalone runnable integration tests for ChatOpenAI with logging (no assertions, no mocks)."""

from __future__ import annotations
import json
import logging
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

# =====================
# LOGGING CONFIG
# =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_TOKEN_COUNT = 100
BASE_URL = "http://shawn-pc.local:8080/v1"


# =====================
# TYPES
# =====================

class ChatInvokeResult(TypedDict):
    success: bool
    content_type: str
    content_preview: str
    model_name: str


class ChatGenerateResult(TypedDict):
    success: bool
    generations: int
    llm_output_keys: list[str]


# =====================
# TESTS AS FUNCTIONS
# =====================

def run_chat_openai_basic() -> ChatInvokeResult:
    """Test ChatOpenAI basic invoke and return structured result with logging."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, base_url=BASE_URL)
    message = HumanMessage(content="Hello, how are you?")
    response = chat.invoke([message])

    success = isinstance(response, BaseMessage)
    logger.info(f"[Basic Invoke] Response type valid: {success}")
    logger.info(f"[Basic Invoke] Model: {chat.model_name}")
    logger.info(f"[Basic Invoke] Content preview: {response.content[:60]}")

    return ChatInvokeResult(
        success=success,
        content_type=type(response.content).__name__,
        content_preview=response.content[:60],
        model_name=chat.model_name,
    )


def run_chat_openai_generate() -> ChatGenerateResult:
    """Test ChatOpenAI generate call with logging."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, n=2, base_url=BASE_URL)
    message = HumanMessage(content="List 3 famous programming languages.")
    result = chat.generate([[message], [message]])

    valid = isinstance(result, LLMResult)
    gens = len(result.generations)
    logger.info(f"[Generate] Result type valid: {valid}")
    logger.info(f"[Generate] Generations count: {gens}")
    logger.info(f"[Generate] LLM output keys: {list(result.llm_output.keys()) if result.llm_output else []}")

    return ChatGenerateResult(
        success=valid,
        generations=gens,
        llm_output_keys=list(result.llm_output.keys()) if result.llm_output else [],
    )


def run_chat_openai_streaming() -> dict[str, Any]:
    """Stream ChatOpenAI output and log each chunk content in real time."""
    chat = ChatOpenAI(
        streaming=True,
        temperature=0.2,
        max_tokens=MAX_TOKEN_COUNT,
        base_url=BASE_URL,
    )

    message = HumanMessage(content="Write a haiku about autumn leaves.")
    logger.info("[Streaming] Starting ChatOpenAI stream...")

    chunks: list[str] = []
    try:
        for chunk in chat.stream([message]):
            if isinstance(chunk, BaseMessageChunk):
                text_piece = getattr(chunk, "content", "")
                if text_piece:
                    chunks.append(text_piece)
                    logger.info(f"[Streaming Chunk] {text_piece.strip()}")
            else:
                logger.debug(f"[Streaming Unknown Chunk Type] {type(chunk)}")
    except Exception as e:
        logger.exception(f"[Streaming] Error during stream: {e}")
        return {"success": False, "stream_count": 0, "content_preview": ""}

    final_response = "".join(chunks)
    logger.info("[Streaming] Completed streaming.")
    logger.info(f"[Streaming] Total chunks received: {len(chunks)}")
    logger.info(f"[Streaming] Combined content: {final_response[:200]}")

    return {
        "success": len(chunks) > 0,
        "stream_count": len(chunks),
        "content_preview": final_response[:200],
    }


# =====================
# MAIN EXECUTION
# =====================

if __name__ == "__main__":
    print("Running ChatOpenAI integration tests manually...\n")

    tests = [
        ("Basic Invoke", run_chat_openai_basic),
        ("Generate", run_chat_openai_generate),
        ("Streaming", run_chat_openai_streaming),
    ]

    for name, fn in tests:
        try:
            result = fn()
            print(f"✅ {name} COMPLETED → {json.dumps(result, indent=2)}\n")
        except Exception as e:
            logger.exception(f"❌ {name} ERROR: {e}")
