from jet.logger import logger  # or just use print if logger not available
from litellm import completion
from typing import Any, Dict

from swarms.structs.agent import LiteLLM


# --- Define a simple tool ---
def get_weather(location: str) -> Dict[str, Any]:
    """Dummy tool to simulate weather info."""
    return {"location": location, "temperature": "30°C", "condition": "Sunny"}


# Tool schema in OpenAI-style function format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g., Manila, Philippines",
                    }
                },
                "required": ["location"],
            },
        },
    }
]


def main():
    # Point LiteLLM at Ollama (ensure Ollama is running: `ollama serve`)
    llm = LiteLLM(
        model_name="ollama/llama3.2",   # Ollama model
        base_url="http://localhost:11434",  # Default Ollama API endpoint
        system_prompt="You are a helpful assistant that can call tools.",
        tools_list_dictionary=tools,   # Register tool definitions
        tool_choice="auto",            # Let the model decide
        verbose=True,
    )

    # User asks a weather question
    task = "What's the weather in Manila right now?"

    response = llm.run(task)

    # If tool call returned
    if isinstance(response, dict) and response.get("function"):
        func = response["function"]["name"]
        args = response["function"]["arguments"]
        logger.info(f"Tool requested: {func} with args: {args}")

        if func == "get_weather":
            weather = get_weather(**eval(args))
            logger.info(f"Tool result: {weather}")
        else:
            logger.warning("Unknown tool call")

    else:
        logger.info(f"LLM direct response: {response}")


if __name__ == "__main__":
    main()
