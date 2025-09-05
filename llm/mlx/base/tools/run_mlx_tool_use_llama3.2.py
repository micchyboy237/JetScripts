from mlx_lm import load, generate
import json

model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(model_id)


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass


conversation = [
    {"role": "user", "content": "What's the weather like in Paris?"}
]
tools = [get_current_weather]

# Format and tokenize the tool use prompt
prompt = tokenizer.apply_chat_template(
    conversation,
    tools=tools,
    add_generation_prompt=True
)

# Generate response
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1000,
)

print(output)
