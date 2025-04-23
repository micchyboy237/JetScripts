from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Generate text
outputs = model.generate(**inputs, max_length=50)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))