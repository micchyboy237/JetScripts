import ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "What's the weather in Nairobi?"}],
    tools=tools,
)

tool_calls = response.message.tool_calls   # This is a list of ToolCall objects
print("Tool Calls:", tool_calls)

if tool_calls:
    call = tool_calls[0]
    # Access attributes properly
    tool_name = call.function.name
    tool_args = call.function.arguments

    print("Tool name:", tool_name)
    print("Arguments:", tool_args)

    # Mock a real result
    result = {
        "role": "tool",
        "name": tool_name,
        "content": "Sunny, 24°C"
    }

    follow_up_stream = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {"role": "user", "content": "What's the weather in Nairobi?"},
            result
        ]
    )
    follow_up = list(follow_up_stream)[-1]

    print("Final response:", follow_up.message.content)
