from llama_cpp import Llama

# Path to the TinyLlama GGUF model (download from Hugging Face)
MODEL_PATH = "./tinyllama-1.1b-chat-v1.0.q4_0.gguf"

# Initialize the model
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=4)

# Generate text
prompt = "Once upon a time"
output = llm(prompt, max_tokens=128, echo=False)

# Print the generated text
print(output["choices"][0]["text"])