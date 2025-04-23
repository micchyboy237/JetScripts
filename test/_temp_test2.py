import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_next_sentence(prompt, max_length=50, temperature=0.7):
    # Check if MPS is available
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the next sentence with optimized settings
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Optimize for memory
        no_repeat_ngram_size=2,  # Prevent repetitive phrases
        top_p=0.9,  # Use nucleus sampling for better diversity
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split into sentences and return the first new sentence after the prompt
    sentences = generated_text[len(prompt):].strip().split('.')
    next_sentence = sentences[0].strip() + '.' if sentences else ""

    return next_sentence


# Example usage
if __name__ == "__main__":
    prompt = "The sun began to set over the calm valley."
    next_sentence = generate_next_sentence(prompt)
    print(f"Prompt: {prompt}")
    print(f"Next sentence: {next_sentence}")
