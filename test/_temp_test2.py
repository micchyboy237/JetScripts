from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Cosine similarity function
def cosine_similarity(embeddings1, embeddings2):
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)


# Function to compute embeddings
def get_embeddings(texts, model, tokenizer, use_mean_pooling=True):
    # Tokenize
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Return embeddings
    if use_mean_pooling:
        return mean_pooling(model_output, encoded_input['attention_mask'])
    else:
        return model_output[0][:, 0, :]  # CLS token embedding


# Example 1: Product Description vs. Customer Review
def example_product_review(model, tokenizer):
    print("\nExample 1: Product Description vs. Customer Review")
    texts = [
        "The EcoSmart Wireless Speaker is a premium audio device featuring Bluetooth 5.0 connectivity, 20W stereo sound, and a 12-hour battery life. Designed with a sleek, portable aluminum body, it’s perfect for outdoor adventures or home use. It supports voice assistants and has an IPX6 water resistance rating, making it ideal for all environments.",
        "I bought the EcoSmart Speaker for a camping trip, and it was fantastic! The sound quality is crisp, and it lasted the whole weekend without needing a recharge. The Bluetooth connected easily, and I love the sturdy design. Only downside is the voice assistant sometimes lags, but it’s great for music."
    ]

    # Compute embeddings with mean pooling
    embeddings_mean = get_embeddings(
        texts, model, tokenizer, use_mean_pooling=True)
    sim_mean = cosine_similarity(
        embeddings_mean[0:1], embeddings_mean[1:2]).item()

    # Compute embeddings without mean pooling (CLS token)
    embeddings_cls = get_embeddings(
        texts, model, tokenizer, use_mean_pooling=False)
    sim_cls = cosine_similarity(
        embeddings_cls[0:1], embeddings_cls[1:2]).item()

    print(f"Cosine Similarity (Mean Pooling): {sim_mean:.4f}")
    print(f"Cosine Similarity (CLS Token): {sim_cls:.4f}")


# Example 2: News Article vs. Summary
def example_news_summary(model, tokenizer):
    print("\nExample 2: News Article vs. Summary")
    texts = [
        "In a groundbreaking discovery, scientists at the European Space Agency have identified a new exoplanet, dubbed Proxima Centauri d, orbiting the star Proxima Centauri, the closest known star to our solar system. The planet, roughly half the mass of Earth, completes an orbit every 5.1 days and lies in the star’s habitable zone, raising hopes for potential life. The discovery was made using the Very Large Telescope in Chile, which detected subtle wobbles in the star’s motion caused by the planet’s gravitational pull. This finding adds to the growing list of exoplanets and fuels excitement for future missions to study their atmospheres.",
        "Scientists from the European Space Agency discovered Proxima Centauri d, a small exoplanet orbiting the nearest star to our solar system. The planet, located in the habitable zone, was detected using the Very Large Telescope, marking a significant step in the search for extraterrestrial life."
    ]

    # Compute embeddings with mean pooling
    embeddings_mean = get_embeddings(
        texts, model, tokenizer, use_mean_pooling=True)
    sim_mean = cosine_similarity(
        embeddings_mean[0:1], embeddings_mean[1:2]).item()

    # Compute embeddings without mean pooling (CLS token)
    embeddings_cls = get_embeddings(
        texts, model, tokenizer, use_mean_pooling=False)
    sim_cls = cosine_similarity(
        embeddings_cls[0:1], embeddings_cls[1:2]).item()

    print(f"Cosine Similarity (Mean Pooling): {sim_mean:.4f}")
    print(f"Cosine Similarity (CLS Token): {sim_cls:.4f}")


# Main function
def main():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    model = AutoModel.from_pretrained(
        'sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

    # Run examples
    example_product_review(model, tokenizer)
    example_news_summary(model, tokenizer)


if __name__ == "__main__":
    main()
