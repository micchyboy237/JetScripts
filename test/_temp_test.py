import mlx.core as mx
import mlx_lm
from mlx_lm import load, generate
from mlx_lm.generate import generate_step

# Load the model and tokenizer
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Define the question
question = "Did Albert Einstein develop the theory of relativity?"

# Format the prompt using the chat template
messages = [
    {"role": "system", "content": "Answer the following question with only 'yes' or 'no'."},
    {"role": "user", "content": question}
]
formatted_prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True)

# Method 1: High-level generate with prompt engineering
print("Method 1: Using generate")
response = generate(
    model,
    tokenizer,
    prompt=formatted_prompt,
    max_tokens=1,  # Limit to one token
    stop=["\n", "<|endoftext|>"],  # Stop at newline or EOS
    temp=0.1,  # Low temperature for deterministic output
    top_p=0.1,  # Low top_p for focused sampling
    verbose=True
)
print(f"Question: {question}")
print(f"Answer: {response}")

# Method 2: Low-level generate_step with logits processor
print("\nMethod 2: Using generate_step with logits processor")

# Tokenize the prompt
input_ids = tokenizer([formatted_prompt], return_tensors="np")["input_ids"]
input_ids = mx.array(input_ids)

# Get token IDs for "yes" and "no"
yes_token = tokenizer.encode("yes")[1]  # Skip BOS token
no_token = tokenizer.encode("no")[1]
allowed_tokens = [yes_token, no_token]
print(f"Yes token ID: {yes_token}, No token ID: {no_token}")

# Custom logits processor to restrict to "yes" or "no"


def logits_processor(logits):
    # Zero out logits for all tokens except "yes" and "no"
    mask = mx.zeros_like(logits)
    mask[:, :, allowed_tokens] = 1
    return logits * mask - (1 - mask) * 1e9  # Suppress non-allowed tokens


# Generate one token
generated_ids = input_ids.tolist()[0]
cache = None
logits, cache = generate_step(
    model,
    input_ids,
    cache=cache,
    temp=0.1,
    top_p=0.1
)
logits = logits_processor(logits)  # Apply restriction
next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
generated_ids.append(next_token)

# Decode the answer
answer = tokenizer.decode([next_token])
print(f"Question: {question}")
print(f"Answer: {answer}")

# Validate the output
if answer.lower() not in ["yes", "no"]:
    print("Warning: Output is not 'yes' or 'no'. Consider adjusting the prompt or logits processor.")
else:
    print("Output is valid.")
