from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: LLMModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


def generate_with_confidence(prompt: str, model, tokenizer, temp=0.8, top_p=0.9, max_tokens=10):
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

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

        token_text = tokenizer.decode([next_token.item()])
        yield token_text, prob


if __name__ == "__main__":
    print("\nWith top_p=0.1:")
    for token, prob in generate_with_confidence("The sky is", model, tokenizer, temp=0.8, top_p=0.1):
        print(f"Token: {token}, Confidence: {prob:.4f}")

    print("\nWith top_p=0.9:")
    for token, prob in generate_with_confidence("The sky is", model, tokenizer, temp=0.8, top_p=0.9):
        print(f"Token: {token}, Confidence: {prob:.4f}")
