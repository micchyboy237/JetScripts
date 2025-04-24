from llama_cpp import Llama

# Load the GGUF model (make sure this path is correct)
model_path = "/Users/jethroestrada/Downloads/T5_Paraphrase_Paws-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=512)

# Input text to be paraphrased
input_text = "The quick brown fox jumps over the lazy dog."

# Prompt format (assumes model expects T5-style input)
prompt = f"paraphrase: {input_text} </s>"

# Run inference
response = llm(prompt, max_tokens=100, stop=["</s>"])
output = response['choices'][0]['text'].strip()

print("Original:", input_text)
print("Paraphrased:", output)
