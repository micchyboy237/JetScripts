from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load, generate
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: ModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


def get_top_n_next_words(prompt, model, tokenizer, n=5, temp=0.8):
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

    # Apply temperature and compute probabilities
    scaled_logits = next_token_logits / temp
    probs = nn.softmax(scaled_logits, axis=-1)

    # Get top n tokens and their probabilities
    top_n_probs, top_n_indices = mx.topk(probs, k=n, axis=-1)
    top_n_probs = top_n_probs[0].tolist()  # Convert to list
    top_n_indices = top_n_indices[0].tolist()

    # Decode tokens to words
    top_n_words = [tokenizer.decode([idx]) for idx in top_n_indices]

    # Return list of (word, probability) tuples
    return list(zip(top_n_words, top_n_probs))


def generate_with_confidence(input_ids, model, tokenizer, temp=0.8, top_p=0.9, max_tokens=10):
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        scaled_logits = next_token_logits / temp
        sorted_logits = mx.sort(scaled_logits, axis=-1)[:, ::-1]
        sorted_probs = nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        mask = cumulative_probs < top_p
        filtered_probs = sorted_probs * mask

        prob_sum = mx.sum(filtered_probs, axis=-1, keepdims=True)
        if prob_sum.item() > 0:
            sorted_probs = filtered_probs / prob_sum
        else:
            sorted_probs = nn.softmax(scaled_logits, axis=-1)

        next_token = mx.random.categorical(sorted_probs)
        prob = sorted_probs[0, next_token.item()].item()

        input_ids = mx.concatenate([input_ids, next_token[None]], axis=-1)
        mx.eval(input_ids)

        token_text = tokenizer.decode([next_token.item()])
        yield token_text, prob


def custom_generate(model, tokenizer, prompt, max_tokens=100, temp=0.8):
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    mx.eval(input_ids)

    generated_tokens = []
    probabilities = []

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        probs = nn.softmax(next_token_logits / temp, axis=-1)
        next_token = mx.argmax(probs, axis=-1)

        generated_tokens.append(next_token.item())
        probabilities.append(probs[0, next_token.item()].item())

        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=-1)
        mx.eval(input_ids)

        token_text = tokenizer.decode([next_token.item()])
        yield token_text, probabilities[-1]

    return generated_tokens, probabilities


if __name__ == "__main__":
    # Example usage for custom_generate
    prompt = "Write a story about Einstein"
    print("Custom Generate Output:")
    for token_text, prob in custom_generate(model, tokenizer, prompt):
        print(f"Token: {token_text}, Confidence: {prob:.4f}")

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

    # Test with different top_p values
    print("\nWith top_p=0.1:")
    for token, prob in generate_with_confidence(input_ids, model, tokenizer, temp=0.8, top_p=0.1):
        print(f"Token: {token}, Confidence: {prob:.4f}")

    print("\nWith top_p=0.9:")
    for token, prob in generate_with_confidence(input_ids, model, tokenizer, temp=0.8, top_p=0.9):
        print(f"Token: {token}, Confidence: {prob:.4f}")
