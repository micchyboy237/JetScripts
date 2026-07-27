from transformers import AutoTokenizer

# Get the model HF name from config and mapping
hf_model_id = "Qwen/Qwen3.5-2B"

# FIXED: Use slow tokenizer to avoid missing backend dependency
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)

# Sample text
sample_text = "This is a test sentence to count tokens using the tokenizer."

# Tokenize and count tokens
tokens = tokenizer.encode(sample_text)
print(f"Text: {sample_text}")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
