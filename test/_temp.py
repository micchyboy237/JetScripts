import json
from openai import OpenAI
from typing import Any, Dict
from jet.utils.commands import copy_to_clipboard

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

# Define tools schema (single source of truth)
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_two_numbers",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract_two_numbers",
            "description": "Subtract two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

# === Dynamically create callable functions from tool names ===
available_functions: Dict[str, callable] = {}

for tool in tools:
    func_name = tool["function"]["name"]
    if func_name == "add_two_numbers":
        available_functions[func_name] = lambda a, b: int(a) + int(b)
    elif func_name == "subtract_two_numbers":
        available_functions[func_name] = lambda a, b: int(a) - int(b)
    # Add more dynamically as needed, or use a registry pattern for scalability

# === Rest of the logic (unchanged) ===
messages = [{'role': 'user', 'content': 'What is three plus one?'}]
print('Prompt:', messages[0]['content'])

create_kwargs = {
  "messages": messages,
  "stream": True,
  "model": "qwen3-instruct-2507:4b",
  "temperature": 0.0,
  "tool_choice": "auto",
  "tools": tools,
}

# === Replace the entire streaming + tool call handling block ===
stream = client.chat.completions.create(**create_kwargs)
chunks = list(stream)
copy_to_clipboard(chunks)

# --- ACCUMULATE TOOL CALLS FROM STREAMING CHUNKS ---
tool_calls = []
current_tool_call = None
accumulated_args = ""

for chunk in chunks:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta

    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if len(tool_calls) <= idx:
                tool_calls.append({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                })
            tool_call = tool_calls[idx]

            # Accumulate ID
            if tc_delta.id:
                tool_call["id"] += tc_delta.id

            # Accumulate function name
            if tc_delta.function and tc_delta.function.name:
                tool_call["function"]["name"] += tc_delta.function.name

            # Accumulate arguments
            if tc_delta.function and tc_delta.function.arguments:
                tool_call["function"]["arguments"] += tc_delta.function.arguments

# Convert accumulated string arguments to dict
for tc in tool_calls:
    try:
        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError:
        print(f"Invalid JSON in arguments: {tc['function']['arguments']}")
        continue

# --- EXECUTE TOOL CALLS ---
messages.append({'role': 'assistant', 'tool_calls': tool_calls})

if tool_calls:
    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        if function_to_call := available_functions.get(func_name):
            print('Calling function:', func_name)
            arguments = tool_call["function"]["arguments"]
            print('Arguments:', arguments)
            output: Any = function_to_call(**arguments)
            print('Function output:', output)
            messages.append({
                "role": "tool",
                "content": json.dumps({"result": output}),
                "tool_call_id": tool_call["id"],
            })
        else:
            print('Function', func_name, 'not found')

    # Second call to get final response
    final_response = client.chat.completions.create(
        model='llama3.2',
        messages=messages,
    )
    print('Final response:', final_response.choices[0].message.content)
else:
    # Fallback: use last content delta if no tool calls
    content = "".join([
        c.choices[0].delta.content or "" for c in chunks
        if c.choices and c.choices[0].delta.content
    ])
    print('No tool calls. Response:', content)