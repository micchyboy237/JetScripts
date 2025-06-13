from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ-053125")

# Define a real-world prompt for generating a customer support email
prompt = """You are a customer support representative for an online retail company. Write a professional, friendly response to a customer who emailed about a delayed order. The customer's name is Alex Johnson, and the order number is #123456. Explain that the delay is due to high demand and provide an estimated delivery date of next Wednesday."""

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Switches between thinking and non-thinking modes. Default is True.
        enable_thinking=False
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
