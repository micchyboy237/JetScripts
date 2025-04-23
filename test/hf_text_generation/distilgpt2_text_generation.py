from transformers import pipeline

# Initialize the text generation pipeline with DistilGPT-2
generator = pipeline("text-generation", model="distilgpt2", device="mps")

# Generate text
prompt = "Once upon a time"
output = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(output[0]["generated_text"])