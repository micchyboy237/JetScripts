#!/usr/bin/env python3
"""
Basic chat demo using ChatLlamaCpp without tools.
Run directly: python run_chat_llama_cpp.py
"""

import os
import shutil
from typing import List
from langchain_core.messages import HumanMessage, BaseMessage
from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")

def demo_basic_chat() -> str:
    """
    Given: A user wants a clear explanation of a science topic
    When: The message is sent to ChatLlamaCpp
    Then: A complete, streamed response is returned and logged
    """
    # Given
    model = "qwen3-instruct-2507:4b"
    base_url = "http://shawn-pc.local:8080/v1"
    temperature = 0.3
    messages: List[BaseMessage] = [
        HumanMessage(content="Explain photosynthesis in simple terms for a 10-year-old.")
    ]

    logger.info("Starting basic chat demo")

    # When
    llm = ChatLlamaCpp(
        model=model,
        temperature=temperature,
        base_url=base_url,
        verbosity="high",
        verbose=True,
        agent_name="demo_basic",
        logger=logger,
    )

    result = llm.invoke(messages)
    response_text = result.content

    # Then
    logger.success(f"Response received ({len(response_text)} chars)")
    return response_text


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("RUNNING BASIC CHAT DEMO (NO TOOLS)")
    logger.info("="*60 + "\n")
    try:
        final_response = demo_basic_chat()
        logger.debug("\n" + "-"*60)
        logger.debug("FINAL RESPONSE:")
        logger.debug("-"*60)
        logger.success(final_response)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    logger.info("\n" + "="*60)
    logger.info("BASIC DEMO COMPLETE")
    logger.info("="*60 + "\n")