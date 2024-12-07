from litellm import completion
import litellm

# [OPTIONAL] REGISTER MODEL - not all ollama models support function calling, litellm defaults to json mode tool calls if native tool calling not supported.

# litellm.register_model(model_cost={
#                 "ollama_chat/llama3.1": {
#                   "supports_function_calling": true
#                 },
#             })

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather like in Boston today?"}]


response = completion(
    model="ollama_chat/llama3.1",
    messages=messages,
    # tools=tools,
    api_base="http://localhost:11434",
    stream=True
)
for chunk in response:
    print(chunk)
