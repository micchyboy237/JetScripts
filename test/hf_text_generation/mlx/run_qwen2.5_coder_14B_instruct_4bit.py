from mlx_lm import load, generate

# Load the model and tokenizer
model, tokenizer = load("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")

# Define a real-world prompt for generating a customer support email
prompt = """You are a customer support representative for an online retail company. Write a professional, friendly response to a customer who emailed about a delayed order. The customer's name is Alex Johnson, and the order number is #123456. Explain that the delay is due to high demand and provide an estimated delivery date of next Wednesday."""

# Apply chat template if available (for instruction-tuned models)
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True)

# Generate the response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=300,
    temperature=0.7,
    verbose=True
)

# Print the generated response
print("\nGenerated Response:\n")
print(response)
