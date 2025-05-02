from jet.logger import logger
from jet.transformers.formatters import format_json
from mlx_lm import load, generate
from jet.llm.mlx.token_utils import extract_texts


# Example usage
if __name__ == "__main__":
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # Define a real-world prompt for generating a customer support email
    prompt = "You are a customer support representative for an online retail company. Write a professional, friendly response to a customer who emailed about a delayed order. The customer's name is Alex Johnson, and the order number is #123456. Explain that the delay is due to high demand and provide an estimated delivery date of next Wednesday."

    # Load the model and tokenizer
    model, tokenizer = load(model_path)

    result = extract_texts(prompt, tokenizer, max_length=20)
    logger.gray("\nResults:")
    logger.success(format_json(result))
