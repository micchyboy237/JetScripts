from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn

model_path: LLMModelType = "llama-3.2-3b-instruct-4bit"
model, tokenizer = load(resolve_model(model_path))


def custom_generate(model, tokenizer, prompt, choices, max_tokens=100, temp=0.8, bias_strength=10.0):
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    mx.eval(input_ids)
    generated_tokens = []
    probabilities = []

    # Create logit bias for choice tokens
    choice_token_map = {choice: tokenizer.encode(choice, add_special_tokens=False)[
        0] for choice in choices}
    logit_bias = {token_id: bias_strength if choice ==
                  "Carbon" else 0.0 for choice, token_id in choice_token_map.items()}

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # Apply logit bias
        for token_id, bias in logit_bias.items():
            next_token_logits[0, token_id] += bias

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
    system_prompt = (
        "Answer the following question by choosing one of the options provided without any additional text.\n"
        "Options:\nOxygen\nCarbon\nNitrogen\nHydrogen"
    )
    question = "Which element is known as the building block of life?"
    choices = ["Oxygen", "Carbon", "Nitrogen", "Hydrogen"]

    # Use chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    print("Custom Generate Output:")
    for token_text, prob in custom_generate(model, tokenizer, prompt, choices, max_tokens=10, temp=0.01):
        print(f"Token: {token_text}, Confidence: {prob:.4f}")
