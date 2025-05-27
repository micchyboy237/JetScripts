from jet.llm.mlx.helpers import load_model
from jet.llm.mlx.mlx_types import LLMModelType, MLXTokenizer
from jet.llm.mlx.models import resolve_model
from mlx_lm import load, generate
from transformers import PreTrainedTokenizer
import mlx.core as mx
import mlx.nn as nn

# Load model and tokenizer
model_path: LLMModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load_model(model_path)


def custom_generate(model: nn.Module, tokenizer: MLXTokenizer, prompt, max_tokens=100, temp=0.8, logit_bias=None):
    # Tokenize the prompt
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(
        1, -1)  # Ensure correct shape
    mx.eval(input_ids)

    generated_tokens = []
    probabilities = []
    active_bias = logit_bias.copy() if logit_bias else {}  # Track active biases
    biased_tokens = set(
        logit_bias.keys()) if logit_bias else set()  # Tokens to bias

    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # Apply logit bias if active
        if active_bias:
            for token_id, bias in active_bias.items():
                next_token_logits[0, token_id] += bias

        # Apply softmax to get probabilities
        probs = nn.softmax(next_token_logits / temp, axis=-1)
        # Greedy sampling
        next_token = mx.argmax(probs, axis=-1)

        # Store token and its probability
        generated_token = next_token.item()
        generated_tokens.append(generated_token)
        probabilities.append(probs[0, generated_token].item())

        # Remove bias for this token if it was generated
        if generated_token in active_bias:
            del active_bias[generated_token]
            biased_tokens.discard(generated_token)

        # Update input_ids
        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=-1)
        mx.eval(input_ids)

        # Decode and yield the token
        token_text = tokenizer.decode(
            [generated_token], skip_special_tokens=False)
        yield token_text, probabilities[-1]

    return generated_tokens, probabilities


prompt_template = """
System: {system_prompt}
Question: {question}
"""
if __name__ == "__main__":
    system_prompt = (
        "Answer the following question by providing only the name of the element that answers the question, "
        "choosing one of the options provided. Do not include any additional text.\n"
        "Options:\nOxygen\nCarbon\nNitrogen\nHydrogen"
    )
    question = "Which element is known as the building block of life?"
    print("Custom Generate Output:")
    prompt = prompt_template.format(
        system_prompt=system_prompt,
        question=question,
    )

    # Example logit_bias: bias towards "Carbon" (token ID assumed, adjust as needed)
    # Simplified, adjust based on actual tokenizer
    carbon_token_id = tokenizer.encode("Carbon", add_special_tokens=False)[0]
    logit_bias = {carbon_token_id: 10.0}  # Bias "Carbon" with +10 logits
    for token_text, prob in custom_generate(model, tokenizer, prompt, logit_bias=logit_bias):
        print(f"Token: {token_text}, Confidence: {prob:.4f}")
