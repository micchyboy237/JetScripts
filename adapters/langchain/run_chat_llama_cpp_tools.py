#!/usr/bin/env python3
"""
Tool-calling chat demo using ChatLlamaCpp.
Run directly: python run_chat_llama_cpp_tools.py
"""

import json
import os
import shutil
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
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

def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def demo_tool_chat() -> Dict[str, Any]:
    """
    Given: A user asks for multiplication and parity check; multiply_numbers tool is available
    When: The message is sent with tools, tool is called, result is fed back via ToolMessage
    Then: Model returns final answer confirming result and parity
    """
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
    llm = ChatLlamaCpp(
        model=model,
        temperature=temperature,
        base_url=base_url,
        verbosity="high",
        verbose=True,
        agent_name="demo_tools",
        logger=logger,
    )

    # First LLM call: get tool request
    result = llm.invoke(messages, tools=tools)
    response_message = result
    tool_calls = getattr(response_message, "tool_call_chunks", None)

    if not tool_calls:
        logger.warning("No tool calls made.")
        return {
            "content": response_message.content,
            "tool_calls": None,
            "tool_result": None,
            "final_response": response_message.content,
            "has_final_answer": True,
        }

    # Extract first (and only) tool call
    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    args_str = tool_call.get("args", "{}")
    tool_call_id = tool_call.get("id")

    logger.success(f"Tool call detected: {tool_name}")
    logger.debug(f"Raw args: {args_str}")

    # Parse and execute tool
    try:
        args = json.loads(args_str)
        a, b = args["a"], args["b"]
        tool_result = multiply_numbers(a, b)
        result_str = str(tool_result)
        logger.success(f"Tool executed: {a} * {b} = {tool_result}")
    except (json.JSONDecodeError, KeyError, Exception) as e:
        result_str = f"Error executing tool: {e}"
        tool_result = None
        logger.error(f"Tool execution failed: {e}")

    # Append ToolMessage and make second LLM call
    messages.append(response_message)  # AI message with tool call
    messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))

    logger.info("Sending tool result back to model for final answer")
    final_result = llm.invoke(messages, tools=tools)
    final_content = final_result.content.strip()

    logger.success("Final answer received from model")

    return {
        "content": response_message.content,
        "tool_calls": tool_calls,
        "tool_name": tool_name,
        "tool_args": args if 'args' in locals() else None,
        "tool_result": tool_result,
        "final_response": final_content,
        "has_final_answer": bool(final_content),
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

        # Updated logic: use tool_result presence instead of has_tool_call
        if result["tool_calls"]:
            for i, call in enumerate(result["tool_calls"]):
                logger.debug(f"Tool Call {i+1}:")
                logger.success(f" Name: {call.get('name')}")
                logger.success(f" Args: {call.get('args')}")
            logger.success(f"Tool Result: {result['tool_result']}")
        else:
            logger.warning("No tool calls made.")
            logger.warning(f"Direct response: {result['content']}")

        # Always show final response
        logger.debug("\n" + "-"*40)
        logger.debug("FINAL MODEL ANSWER:")
        logger.debug("-"*40)
        logger.success(result["final_response"])

    except Exception as e:
        logger.error(f"Demo failed: {e}")
    logger.info("\n" + "="*60)
    logger.info("TOOL DEMO COMPLETE")
    logger.info("="*60 + "\n")