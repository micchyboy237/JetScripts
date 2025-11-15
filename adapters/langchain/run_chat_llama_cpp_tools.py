#!/usr/bin/env python3
"""
Tool-calling chat demo using ChatLlamaCpp.
Run directly: python run_chat_llama_cpp_tools.py
"""

import os
import shutil
from typing import List, Any
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

def demo_tool_chat() -> dict[str, Any]:
    """
    Given: A user asks for weather and a calculator tool is available
    When: The message is sent with tools enabled
    Then: The model calls the tool correctly and streams args
    """
    # Given
    model = "qwen3-instruct-2507:4b"
    base_url = "http://shawn-pc.local:8080/v1"
    temperature = 0.0
    messages: List[BaseMessage] = [
        HumanMessage(content="What is 58 * 47? Then tell me if it's even or odd.")
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply_numbers",
                "description": "Multiply two integers and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    logger.info("Starting tool-calling chat demo")

    # When
    llm = ChatLlamaCpp(
        model=model,
        temperature=temperature,
        base_url=base_url,
        verbosity="high",
        verbose=True,
        agent_name="demo_tools",
        logger=logger,
    )

    result = llm.invoke(messages, tools=tools)
    response_message = result

    # Then
    tool_calls = getattr(response_message, "tool_call_chunks", None)
    content = response_message.content

    logger.success("Tool call demo completed")
    return {
        "content": content,
        "tool_calls": tool_calls,
        "has_tool_call": bool(tool_calls),
    }


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("RUNNING TOOL-CALLING CHAT DEMO")
    logger.info("="*60 + "\n")
    try:
        result = demo_tool_chat()
        logger.gray("\n" + "-"*60)
        logger.gray("TOOL CALL RESULT:")
        logger.gray("-"*60)
        if result["has_tool_call"]:
            for i, call in enumerate(result["tool_calls"]):
                logger.debug(f"Tool Call {i+1}:")
                logger.sucess(f"  Name: {call.get('name')}")
                logger.sucess(f"  Args: {call.get('args')}")
        else:
            logger.warning("No tool calls made.")
            logger.warning(f"Direct response: {result['content']}")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    logger.info("\n" + "="*60)
    logger.info("TOOL DEMO COMPLETE")
    logger.info("="*60 + "\n")