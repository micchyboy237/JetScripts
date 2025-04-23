from llama_cpp import Llama

# Path to the Grok Mini GGUF model (download from Hugging Face community uploads)
MODEL_PATH = "./grok-mini-3b.q4_0.gguf"

# Initialize the model
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=4)

# Generate text
prompt = "Once upon a time"
output = llm(prompt, max_tokens=100, echo=False)

# Print the generated text
print(output["choices"][0]["text"])