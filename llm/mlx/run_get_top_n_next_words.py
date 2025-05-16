from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: ModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


def get_top_n_next_words(prompt, model, tokenizer, n=5, temp=1.0):
    """
    Get the top n next words and their probabilities for a given prompt.

    Args:
        prompt (str): The input partial sentence.
        model: The MLX model.
        tokenizer: The tokenizer.
        n (int): Number of top words to return.
        temp (float): Temperature for softmax.

    Returns:
        List[Tuple[str, float]]: List of (word, probability) tuples for the top n next words.
    """
    # Tokenize the prompt
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    mx.eval(input_ids)

    # Forward pass
    logits = model(input_ids)
    next_token_logits = logits[:, -1, :]  # Logits for the next token

    # Debug: Inspect logits
    logit_max = mx.max(next_token_logits).item()
    logit_min = mx.min(next_token_logits).item()
    print(f"Logits range: min={logit_min:.4f}, max={logit_max:.4f}")

    # Clip logits to prevent numerical issues
    next_token_logits = mx.clip(next_token_logits, -1e9, 1e9)

    # Apply temperature and compute probabilities
    scaled_logits = next_token_logits / temp
    probs = nn.softmax(scaled_logits, axis=-1)

    # Debug: Check if probs are valid
    prob_sum = mx.sum(probs).item()
    if not (0.99 <= prob_sum <= 1.01):
        print(f"Warning: Probability sum is {prob_sum}, expected ~1.0")

    # Debug: Inspect top raw probabilities
    top_raw_probs = mx.sort(probs[0])[-10:][::-1].tolist()
    print(f"Top 10 raw probabilities: {top_raw_probs}")

    # Get top n indices using argsort (descending order)
    # Negate probs for descending order
    top_n_indices = mx.argsort(-probs[0])[:n]
    top_n_indices = top_n_indices.astype(mx.int32)  # Cast to integral dtype
    top_n_indices = top_n_indices.tolist()

    # Get corresponding probabilities
    top_n_probs = probs[0, top_n_indices].tolist()

    # Debug: Print raw indices and probabilities
    print(f"Top {n} indices: {top_n_indices}")
    print(f"Top {n} probabilities: {top_n_probs}")

    # Decode tokens to words, skipping special tokens
    top_n_words = []
    for idx in top_n_indices:
        decoded = tokenizer.decode([idx])
        # Use placeholder for empty or special tokens
        if not decoded.strip() or idx == 0:  # Skip index 0 (likely special token)
            decoded = f"<token_{idx}>"
        top_n_words.append(decoded)

    # Return list of (word, probability) tuples
    return list(zip(top_n_words, top_n_probs))


if __name__ == "__main__":
    # Example usage for get_top_n_next_words
    prompt = "The sky is"
    print("\nTop 5 next words for prompt 'The sky is':")
    top_n_results = get_top_n_next_words(
        prompt, model, tokenizer, n=5, temp=0.8)
    for word, prob in top_n_results:
        print(f"Word: {word}, Probability: {prob:.4f}")

    # Prepare input_ids for generate_with_confidence
    input_ids = mx.array(tokenizer.encode("The sky is")).reshape(1, -1)
    mx.eval(input_ids)
