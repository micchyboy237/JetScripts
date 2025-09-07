import sys

from mlx_lm import load, stream_generate
from typing import List, Dict, Any

from jet.llm.mlx.mlx_utils import get_chat_template

def create_chat_prompt(tokenizer: Any, system_prompt: str, user_message: str) -> str:
    """Creates a formatted chat prompt with system and user messages."""
    # Custom chat template to handle system prompt
    custom_template = get_chat_template("mlx-community/Llama-3.2-3B-Instruct-4bit")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return tokenizer.apply_chat_template(
        messages,
        chat_template=custom_template,
        tokenize=False,
        add_generation_prompt=True
    )

def stream_chat_response(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 100) -> None:
    """Streams the chat response token by token."""
    for token in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        print(token.text, end="", flush=True)
    print()  # Newline after streaming completes

def main():
    # Model configuration
    model_name = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    system_prompt = "You are a helpful AI assistant with expertise in programming and general knowledge."
    user_message = "How do I write a Python function to reverse a string?"
    max_tokens = 150

    # Load model and tokenizer
    try:
        model, tokenizer = load(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create prompt
    prompt = create_chat_prompt(tokenizer, system_prompt, user_message)

    # Stream response
    print("\nAssistant: ")
    stream_chat_response(model, tokenizer, prompt, max_tokens)

if __name__ == "__main__":
    main()