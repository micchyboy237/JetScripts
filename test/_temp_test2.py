import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_next_sentences(prompt, num_sentences=1, max_length_per_sentence=50, temperature=0.7):
    """
    Generate `num_sentences` sentences following the input prompt.

    Args:
        prompt (str): Input prompt text.
        num_sentences (int): Number of sentences to generate.
        max_length_per_sentence (int): Approximate max length per sentence.
        temperature (float): Sampling temperature for generation.

    Returns:
        str: Generated sentences joined together.
    """
    # Check if MPS is available
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Set pad token to avoid attention mask warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Fallback to eos_token

    # Encode the input prompt
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # Explicitly create attention mask
    attention_mask = inputs["attention_mask"].to(device)

    # Calculate max_length to accommodate prompt + n sentences
    max_length = len(inputs["input_ids"][0]) + \
        (max_length_per_sentence * num_sentences)

    # Generate text with optimized settings
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_p=0.9,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract text after the prompt
    generated_after_prompt = generated_text[len(prompt):].strip()

    # Split into sentences
    sentences = [s.strip()
                 for s in generated_after_prompt.split(".") if s.strip()]

    # Collect up to num_sentences, ensuring each ends with a period
    selected_sentences = []
    for i in range(min(num_sentences, len(sentences))):
        sentence = sentences[i]
        if not sentence.endswith("."):
            sentence += "."
        selected_sentences.append(sentence)

    # If fewer sentences than requested, pad with a generic sentence
    while len(selected_sentences) < num_sentences:
        selected_sentences.append("The story continues.")

    # Join the sentences with a space
    return " ".join(selected_sentences)


# Example usage
if __name__ == "__main__":
    prompt = "The sun began to set over the calm valley."
    num_sentences = 3  # Number of sentences to generate
    generated_sentences = generate_next_sentences(
        prompt, num_sentences=num_sentences)
    print(f"Prompt: {prompt}")
    print(f"Generated sentences: {generated_sentences}")
