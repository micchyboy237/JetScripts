"""
Streaming tool-calling chat demo using ChatLlamaCpp with llm.stream().
Run directly: python run_chat_llama_cpp_tools_stream.py
"""
import json
import os
import shutil
from typing import List, Any, Dict, Optional
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessageChunk
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


def demo_tool_chat_stream() -> Dict[str, Any]:
    """
    Given: User asks for multiplication and parity; multiply_numbers tool is available
    When: Stream first response → detect tool call → execute tool → stream final answer
    Then: All output (tool args, result, final answer) is printed in real time
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

    logger.info("Starting streaming tool-calling demo")
    llm = ChatLlamaCpp(
        model=model,
        temperature=temperature,
        base_url=base_url,
        verbosity="high",
        verbose=True,
        agent_name="demo_tools_stream",
        logger=logger,
    )

    # === PHASE 1: Stream initial response with tool call ===
    logger.info("Streaming initial response...")
    tool_call_chunks: List[dict] = []
    text_content = ""
    response_message: Optional[AIMessageChunk] = None  # Will hold last chunk for reference

    for chunk in llm.stream(messages, tools=tools):
        msg = chunk.message
        if hasattr(msg, "content_blocks") and msg.content_blocks:
            for block in msg.content_blocks:
                btype = block.get("type")
                if btype == "tool_call_chunk":
                    tool_call_chunks.append(block)
                elif btype == "text" and block.get("text"):
                    text_content += block["text"]
        elif msg.content:
            text_content += msg.content

    if not tool_call_chunks:
        logger.warning("No tool calls detected in stream.")
        final_answer = text_content.strip()
        logger.success(f"Direct streamed answer: {final_answer}")
        return {
            "tool_calls": None,
            "tool_result": None,
            "final_response": final_answer,
            "has_final_answer": True,
        }

    # === Reconstruct tool call from chunks ===
    response_message = None  # Will be rebuilt as full AIMessage
    first_chunk = tool_call_chunks[0]
    tool_name = first_chunk.get("name")
    tool_call_id = first_chunk.get("id")
    args_str = text_content  # args streamed into text_content
    logger.success(f"Tool call streamed: {tool_name}")
    logger.debug(f"Raw streamed args: {args_str}")

    try:
        args = json.loads(args_str)
        a, b = args["a"], args["b"]
        tool_result = multiply_numbers(a, b)
        result_str = str(tool_result)
        logger.success(f"Tool executed: {a} * {b} = {tool_result}")
    except (json.JSONDecodeError, KeyError, Exception) as e:
        result_str = f"Error: {e}"
        tool_result = None
        logger.error(f"Tool execution failed: {e}")

    # === Build full AIMessage from chunks ===
    from langchain_core.messages import AIMessage
    full_ai_message = AIMessage(
        content="" if tool_call_chunks else text_content,
        tool_call_chunks=[{
            "name": tool_name,
            "args": args_str,
            "id": tool_call_id,
            "type": "tool_call"
        }] if tool_call_chunks else None
    )

    # === PHASE 2: Send tool result and stream final answer ===
    messages.append(full_ai_message)
    messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))

    logger.info("Streaming final answer after tool result...")
    final_content = ""
    for chunk in llm.stream(messages, tools=tools):
        if chunk.message.content:
            final_content += chunk.message.content
            # Real-time print is handled by ChatLlamaCpp._stream (teal output)

    final_answer = final_content.strip()
    logger.success("Final answer streamed from model")

    return {
        "tool_calls": tool_call_chunks,
        "tool_name": tool_name,
        "tool_args": args if 'args' in locals() else None,
        "tool_result": tool_result,
        "final_response": final_answer,
        "has_final_answer": bool(final_answer),
    }


if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("RUNNING STREAMING TOOL-CALLING CHAT DEMO")
    logger.info("="*60 + "\n")

    try:
        result = demo_tool_chat_stream()
        logger.gray("\n" + "-"*60)
        logger.gray("STREAMING TOOL CALL RESULT:")
        logger.gray("-"*60)

        if result["tool_calls"]:
            logger.success(f"Tool: {result['tool_name']}")
            logger.success(f"Args: {result['tool_args']}")
            logger.success(f"Result: {result['tool_result']}")
        else:
            logger.warning("No tool calls in stream")

        logger.debug("\n" + "-"*40)
        logger.debug("FINAL STREAMED ANSWER:")
        logger.debug("-"*40)
        logger.success(result["final_response"])

    except Exception as e:
        logger.error(f"Streaming demo failed: {e}")

    logger.info("\n" + "="*60)
    logger.info("STREAMING TOOL DEMO COMPLETE")
    logger.info("="*60 + "\n")