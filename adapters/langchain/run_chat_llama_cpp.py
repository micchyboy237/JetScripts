"""
Fixed ChatLlamaCpp + structured tool calling
"""
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp

log = logging.getLogger(__name__)

class GetWeatherInput(BaseModel):
    """Input for get_weather."""
    location: str = Field(..., description="City name, e.g. 'Paris'")

def get_weather(input: GetWeatherInput) -> str:
    """Mock weather tool."""
    log.info(f">>> [TOOL EXECUTION] get_weather(location='{input.location}')")
    return f"It's sunny and 25°C in {input.location} today!"

llm = ChatLlamaCpp(
    model="qwen3-instruct-2507:4b",
    temperature=0.1,
    base_url="http://shawn-pc.local:8080/v1",
)

llm_with_tools = (
    llm.bind_tools(
        tools=[get_weather],
        tool_choice="any"
    )
    .with_structured_output(
        schema=GetWeatherInput,
        method="json_mode",  # Switch to json_mode for simpler output
        include_raw=True  # Keep for debugging
    )
)

# System message for JSON output
messages = [
    SystemMessage(content="Extract the city name from the user's query and output a JSON object with a 'location' key, e.g., {'location': 'Paris'} for the GetWeatherInput tool. If no city is specified, use {'location': 'unknown'}."),
    HumanMessage(content="What's the weather in Paris?")
]

print("\n=== First LLM Call (Structured Tool Input) ===")
max_retries = 2
for attempt in range(max_retries):
    try:
        result = llm_with_tools.invoke(messages)
        log.debug(f"Raw LLM output (attempt {attempt + 1}): {result.get('raw')}")
        structured = result.get("parsed") or GetWeatherInput(location="unknown")
        print(f"Parsed input: {structured}")
        break
    except Exception as e:
        log.error(f"Attempt {attempt + 1} failed: {e}")
        if attempt == max_retries - 1:
            structured = GetWeatherInput(location="unknown")

tool_result = get_weather(structured)
print(f"Tool result: {tool_result}")

messages.append(ToolMessage(
    content=tool_result,
    tool_call_id="manual_123"
))

print("\n=== Second LLM Call (Summarize) ===")
final = llm.invoke(messages)
print(f"Final AI: {final.content}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("User : What's the weather in Paris?")
print(f"Tool : {structured.location}")
print(f"Result: {tool_result}")
print(f"AI   : {final.content}")
print("="*50)