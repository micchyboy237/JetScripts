from llama_cpp import Llama


def generate_text_stream(model_path, prompt, max_tokens=200, temperature=0.7):
    # Initialize the model
    llm = Llama(model_path=model_path, n_ctx=2048)

    # Prepare the chat format for TinyLlama
    formatted_prompt = f"<|USER|> {prompt} <|ASSISTANT|>"

    # Stream response
    response_stream = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|USER|>", "<|ASSISTANT|>"],
        echo=False,
        stream=True
    )

    print("Generated Response:")
    for chunk in response_stream:
        print(chunk['choices'][0]['text'], end='', flush=True)
    print()  # Newline after stream ends


def main():
    # Model path
    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q5_0.gguf"

    # New Prompt
    prompt = """You are a customer support representative for an online retail company. Write a professional, friendly response to a customer who emailed about a delayed order. The customer's name is Alex Johnson, and the order number is #123456. Explain that the delay is due to high demand and provide an estimated delivery date of next Wednesday."""

    try:
        generate_text_stream(model_path, prompt)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
