from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: ModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


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
