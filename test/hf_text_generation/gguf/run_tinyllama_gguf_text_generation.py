from llama_cpp import Llama


def generate_text(model_path, prompt, max_tokens=200, temperature=0.7):
    # Initialize the model
    llm = Llama(model_path=model_path, n_ctx=2048)

    # Prepare the chat format for TinyLlama
    formatted_prompt = f"<|USER|> {prompt} <|ASSISTANT|>"

    # Generate response
    output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|USER|>", "<|ASSISTANT|>"],
        echo=False
    )

    # Extract and return the generated text
    return output['choices'][0]['text'].strip()


def main():
    # Model path
    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q5_0.gguf"

    # Example prompt
    prompt = "Write a short description of a futuristic city."

    try:
        # Generate text
        response = generate_text(model_path, prompt)
        print("Generated Response:")
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
