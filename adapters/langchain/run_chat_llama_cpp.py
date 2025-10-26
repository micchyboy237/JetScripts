#!/usr/bin/env python
"""
Test: ChatLlamaCpp with structured tool calling using Pydantic + bind_tools + with_structured_output
"""

import logging
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, ToolMessage
from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp

# ========================================
# 1. Setup Logging
# ========================================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ========================================
# 2. Define Tool with Pydantic Schema
# ========================================
class GetWeatherInput(BaseModel):
    """Input schema for get_weather tool."""
    location: str = Field(..., description="The city name to check weather for, e.g., 'Paris'")

def get_weather(input: GetWeatherInput) -> str:
    """Get current weather for a location."""
    log.info(f">>> [TOOL EXECUTION] get_weather(location='{input.location}')")
    return f"It's sunny and 25°C in {input.location} today!"


# ========================================
# 3. Initialize LLM
# ========================================
llm = ChatLlamaCpp(
    model="qwen3-instruct-2507:4b",
    temperature=0.1,
    base_url="http://shawn-pc.local:8080/v1",  # adjust if needed
)

# ========================================
# 4. Build Runnable: bind tool → force structured output
# ========================================
llm_with_tools = (
    llm.bind_tools(
        tools=[get_weather],
        tool_choice="any"  # Force tool usage
    )
    .with_structured_output(
        schema=GetWeatherInput,
        method="function_calling",
        include_raw=False  # Set True if you want raw AIMessage too
    )
)

# ========================================
# 5. Conversation Flow
# ========================================
messages = [HumanMessage(content="What's the weather in Paris?")]

print("\n=== First LLM Call (Structured Tool Input) ===")
try:
    structured_input: GetWeatherInput = llm_with_tools.invoke(messages)
    print(f"Parsed Tool Input: {structured_input}")
except Exception as e:
    log.error(f"Failed to get structured input: {e}")
    structured_input = GetWeatherInput(location="unknown")

# ========================================
# 6. Execute Tool
# ========================================
tool_result = get_weather(structured_input)
print(f"Tool Result: {tool_result}")

# ========================================
# 7. Optional: Let LLM Summarize Result
# ========================================
messages.append(ToolMessage(
    content=tool_result,
    tool_call_id="manual_id_123"  # dummy ID since structured output skips real tool_call
))

print("\n=== Second LLM Call (Summarize Tool Result) ===")
final_response = llm.invoke(messages)  # plain LLM, no tools
print(f"Final AI Response: {final_response.content}")

# ========================================
# 8. Final Output Summary
# ========================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("User: What's the weather in Paris?")
print(f"Tool Input: {structured_input.location}")
print(f"Tool Output: {tool_result}")
print(f"AI: {final_response.content}")
print("="*50)