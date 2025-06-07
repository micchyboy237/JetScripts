from mlx_lm import load
from transformers import AutoTokenizer
import mlx.core as mx
import numpy as np


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens):
    prefix_tokens = list(prefix_tokens)
    suffix_tokens = list(suffix_tokens)
    num_reserved = len(prefix_tokens) + len(suffix_tokens)

    inputs = tokenizer(
        pairs,
        padding=False,
        truncation=True,
        max_length=max_length - num_reserved,
        return_tensors="np"
    )
    input_ids_list = inputs['input_ids'].tolist()

    input_ids = []
    for single_input in input_ids_list:
        combined = prefix_tokens + list(single_input) + suffix_tokens
        if len(combined) > max_length:
            # Truncate the middle part (single_input)
            max_input_len = max_length - \
                len(prefix_tokens) - len(suffix_tokens)
            truncated_input = single_input[:max_input_len]
            combined = prefix_tokens + truncated_input + suffix_tokens
        input_ids.append(combined)

    padded_inputs = tokenizer.pad(
        {'input_ids': input_ids},
        padding='max_length',
        max_length=max_length,
        return_tensors="np"
    )

    input_ids = mx.array(padded_inputs['input_ids'], dtype=mx.int32)
    attention_mask = mx.array(padded_inputs['attention_mask'], dtype=mx.int32)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def compute_logits(model, inputs, token_true_id, token_false_id):
    # Forward pass (no attention_mask for MLX)
    # Shape: [batch_size, seq_len, vocab_size]
    logits = model(inputs['input_ids'])

    # Get logits for last token
    batch_scores = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

    # Extract "yes" and "no" logits
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]

    # Stack: [batch_size, 2]
    stacked = mx.stack([false_vector, true_vector], axis=1)

    # Apply softmax to get probabilities
    probs = mx.softmax(stacked, axis=1)

    # Return "yes" probability (index 1)
    scores = probs[:, 1].tolist()
    return scores


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", padding_side='left')
# MLX model loading
model, _ = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

# Get token IDs for "yes" and "no"
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 1024

# Define prefix and suffix
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(
    prefix, padding=True, add_special_tokens=False)
suffix_tokens = tokenizer.encode(
    suffix, padding=True, add_special_tokens=False)

# Ensure prefix and suffix tokens are Python lists
prefix_tokens = list(prefix_tokens)
suffix_tokens = list(suffix_tokens)

# Define task and inputs
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Format input pairs
pairs = [format_instruction(task, query, doc)
         for query, doc in zip(queries, documents)]

# Process inputs
inputs = process_inputs(pairs, tokenizer, max_length,
                        prefix_tokens, suffix_tokens)

# Compute scores
scores = compute_logits(model, inputs, token_true_id, token_false_id)

print("Scores:", scores)
