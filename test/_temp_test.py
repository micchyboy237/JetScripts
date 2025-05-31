from mlx_lm import load, generate
from mlx_lm.evaluate import MLXLM
import numpy as np
from typing import List, Tuple


class Qwen3Evaluator:
    def __init__(self, model_path: str = "mlx-community/Qwen3-1.7B-4bit-DWQ", max_tokens: int = 512):
        self.model, self.tokenizer = load(model_path)
        self.evaluator = MLXLM(
            model_path, max_tokens=max_tokens, use_chat_template=True)
        self.max_tokens = max_tokens

    def generate_text(self, prompt: str) -> str:
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens)

    def compute_perplexity(self, text: str) -> float:
        request = type("Instance", (), {"args": (text,)})
        log_probs = self.evaluator.loglikelihood_rolling([request])[0]
        expected = float
        result = float(
            np.exp(-log_probs / len(self.evaluator._tokenize([text])[0])))
        assert isinstance(
            result, expected), f"Expected {expected}, but got {type(result)}"
        return result

    def compute_confidence(self, context: str, continuation: str) -> float:
        request = type("Instance", (), {"args": (context, continuation)})
        log_prob, _ = self.evaluator.loglikelihood([request])[0]
        expected = float
        # Convert log probability to probability
        result = float(np.exp(log_prob))
        assert isinstance(
            result, expected), f"Expected {expected}, but got {type(result)}"
        return result


if __name__ == "__main__":
    evaluator = Qwen3Evaluator()

    # Generate text
    prompt = "Write a short poem about the stars."
    generated_text = evaluator.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # Compute perplexity
    perplexity = evaluator.compute_perplexity(generated_text)
    print(f"Perplexity: {perplexity:.2f}")

    # Compute confidence
    context = "The sky is full of"
    continuation = " twinkling stars."
    confidence = evaluator.compute_confidence(context, continuation)
    print(f"Confidence for '{context}' -> '{continuation}': {confidence:.4f}")
