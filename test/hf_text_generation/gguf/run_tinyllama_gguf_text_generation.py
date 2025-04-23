from llama_cpp import Llama

# model_path = "models/tinyllama.gguf"
model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q5_0.gguf"

# Load the TinyLlama model
llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4,  # Adjust based on your CPU
)


# Function for text generation
def generate_text(prompt: str, max_tokens: int = 128, temperature: float = 0.7):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
    )
    return output["choices"][0]["text"]


# Example usage
if __name__ == "__main__":
    prompt = "Explain quantum computing in simple terms:"
    response = generate_text(prompt)
    print(f"\nPrompt:\n{prompt}\n\nResponse:\n{response}")
