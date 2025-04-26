from mlx_lm import load, generate, stream_generate

# Load the model and tokenizer
model, tokenizer = load("mlx-community/gemma-3-1b-it-qat-4bit")

prompt = """
```python
def calculate_area_of_circle(radius):
    \"\"\"Calculate the area of a circle given its radius.\"\"\"
"""


# Apply chat template if available (for instruction-tuned models)
# if tokenizer.chat_template is not None:
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt},
#     ]
#     prompt = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True)


# Generate the response
print("\nGenerate...")
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
)

# Print the generated response
print("\nGenerated Response:\n")
print(response)

# Stream generate the response
print("\nStream generate...")
response_stream = stream_generate(
    model,
    tokenizer,
    prompt=prompt,
)

for response in response_stream:
    print(response.text, end='', flush=True)
