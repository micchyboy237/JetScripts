"""Demonstration of ChatLlamaCpp tool usage with bind_tools()."""

from jet.adapters.langchain.chat_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import json


# ---------------------------------------------------------------------
# 1️⃣ Define tool functions
# ---------------------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get current weather for a given city (mocked)."""
    weather_data = {
        "Tokyo": "Sunny, 25°C",
        "Manila": "Cloudy, 30°C",
        "London": "Rainy, 15°C",
    }
    return weather_data.get(location, "Unknown location.")


@tool
def calculate_sum(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


# ---------------------------------------------------------------------
# 2️⃣ Instantiate ChatLlamaCpp and bind tools
# ---------------------------------------------------------------------
llm = ChatOllama(
    model="llama3.2",  # or any available model in Ollama
    agent_name="tool_demo",
    temperature=0,
)

# Bind the tools
llm_with_tools = llm.bind_tools([get_weather, calculate_sum])


# ---------------------------------------------------------------------
# 3️⃣ Simulate chat and handle tool calls
# ---------------------------------------------------------------------
def run_tool_demo():
    messages = [
        HumanMessage(content="What is the weather in Tokyo?"),
    ]

    # Step 1: Model processes message
    result = llm_with_tools.invoke(messages)
    ai_msg = result if isinstance(result, AIMessage) else result.generations[0].message

    # Step 2: If tool calls detected, execute and return response
    if ai_msg.tool_calls:
        print("\n[Tool Calls Detected]")
        for tool_call in ai_msg.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f" - {name}({args})")

            if name == "get_weather":
                output = get_weather.invoke(args)
            elif name == "calculate_sum":
                output = calculate_sum.invoke(args)
            else:
                output = f"Unknown tool: {name}"

            # Send tool result back to model
            messages.append(ai_msg)
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(output)))

            final_response = llm_with_tools.invoke(messages)
            print("\n[Final Model Response]")
            print(final_response.content)
    else:
        print("\n[Direct Model Response]")
        print(ai_msg.content)


if __name__ == "__main__":
    run_tool_demo()
