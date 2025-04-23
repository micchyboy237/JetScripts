from llama_cpp import Llama

# Path to the Phi-3 Mini GGUF model (download from Hugging Face)
MODEL_PATH = "./phi-3-mini-4k-instruct.q4_0.gguf"

# Initialize the model
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=4)

# Generate text
prompt = "Once upon a time"
output = llm(prompt, max_tokens=100, echo=False)

# Print the generated text
print(output["choices"][0]["text"])