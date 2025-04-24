from llama_cpp import Llama
import re

# Initialize the GGUF model
# Update with your model path
model_path = "~/Downloads/T5_Paraphrase_Paws-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)

# Define the prompt template for paraphrasing
prompt_template = """paraphrase: {input_text}"""

# Input sentence to paraphrase
input_sentence = "The quick brown fox jumps over the lazy dog."

# Create the prompt
prompt = prompt_template.format(input_text=input_sentence)

# Generate the paraphrased output
output = llm(
    prompt,
    max_tokens=100,
    # temperature=0.7,
    # top_p=0.9,
    stop=["\n"],  # Stop at newline to keep output clean
)

# Extract the generated text
paraphrased_text = output["choices"][0]["text"].strip()

# Clean up the output (remove any residual prompt artifacts)
cleaned_text = re.sub(r"^paraphrase:.*", "",
                      paraphrased_text, flags=re.MULTILINE).strip()

# Print the results
print("Original sentence:", input_sentence)
print("Paraphrased sentence:", cleaned_text if cleaned_text else paraphrased_text)

# Free model resources
llm.close()
