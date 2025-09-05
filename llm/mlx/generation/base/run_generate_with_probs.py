from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load, generate
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: LLMModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


def custom_generate(model, tokenizer, prompt, max_tokens=100, temp=0.8):
    # Tokenize the prompt
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(
        1, -1)  # Ensure correct shape
    mx.eval(input_ids)

    generated_tokens = []
    probabilities = []

    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = nn.softmax(next_token_logits / temp, axis=-1)
        # Greedy sampling
        next_token = mx.argmax(probs, axis=-1)

        # Store token and its probability
        generated_tokens.append(next_token.item())
        probabilities.append(probs[0, next_token.item()].item())

        # Update input_ids
        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=-1)
        mx.eval(input_ids)

        # Decode and yield the token
        token_text = tokenizer.decode([next_token.item()])
        yield token_text, probabilities[-1]

    return generated_tokens, probabilities


prompt_template = """
System: {system_prompt}
Question: {question}
"""
if __name__ == "__main__":
    # system_prompt = "Answer the following question by choosing one of the options provided without any additional text.\nOptions:\nOxygen\nCarbon\nNitrogen\nHydrogen"
    system_prompt = (
        "Answer the following question by providing only the name of the element that answers the question, "
        "choosing one of the options provided. Do not include any additional text.\n"
        "Options:\nOxygen\nCarbon\nNitrogen\nHydrogen"
    )
    # Example usage for custom_generate
    question = "Which element is known as the building block of life?"
    print("Custom Generate Output:")
    prompt = prompt_template.format(
        system_prompt=system_prompt, question=question)
    for token_text, prob in custom_generate(model, tokenizer, prompt):
        print(f"Token: {token_text}, Confidence: {prob:.4f}")
