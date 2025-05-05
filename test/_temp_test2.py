from jet.wordnet.sentence import split_sentences
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Cosine similarity function
def cosine_similarity(embeddings1, embeddings2):
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)


# Function to compute embeddings
def get_embeddings(texts, model, tokenizer, use_mean_pooling=True):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    if use_mean_pooling:
        return mean_pooling(model_output, encoded_input['attention_mask'])
    else:
        return model_output[0][:, 0, :]  # CLS token embedding


# Function to perform similarity search
def similarity_search(query, texts, model, tokenizer, use_mean_pooling=True, top_k=5):
    query_embedding = get_embeddings(
        [query], model, tokenizer, use_mean_pooling)
    text_embeddings = get_embeddings(texts, model, tokenizer, use_mean_pooling)
    similarities = cosine_similarity(query_embedding, text_embeddings)
    similarities = similarities.cpu().numpy()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_scores = similarities[top_k_indices]
    results = [(texts[idx], top_k_scores[i])
               for i, idx in enumerate(top_k_indices)]
    return results


def main():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    model = AutoModel.from_pretrained(
        'sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

    # Example content
    content = """## Naruto: Shippuuden Movie 6 - Road to Ninja
Movie, 2012 Finished 1 ep, 109 min
Action Adventure Fantasy
Naruto: Shippuuden Movie 6 - Road to Ninja
Returning home to Konohagakure, the young ninja celebrate defeating a group of supposed Akatsuki members. Naruto Uzumaki and Sakura Haruno, however, feel differently. Naruto is jealous of his comrades' congratulatory families, wishing for the presence of his own parents. Sakura, on the other hand, is angry at her embarrassing parents, and wishes for no parents at all. The two clash over their opposing ideals, but are faced with a more pressing matter when the masked Madara Uchiha suddenly appears and transports them to an alternate world. In this world, Sakura's parents are considered heroes--for they gave their lives to protect Konohagakure from the Nine-Tailed Fox attack 10 years ago. Consequently, Naruto's parents, Minato Namikaze and Kushina Uzumaki, are alive and well. Unable to return home or find the masked Madara, Naruto and Sakura stay in this new world and enjoy the changes they have always longed for. All seems well for the two ninja, until an unexpected threat emerges that pushes Naruto and Sakura to not only fight for the Konohagakure of the alternate world, but also to find a way back to their own. [Written by MAL Rewrite]
Studio Pierrot
Source Manga
Theme Isekai
Demographic Shounen
7.68
366K
Add to My List"""

    # Split content into sentences
    texts = split_sentences(content)

    # Example query
    query = "Naruto and Sakura in an alternate world"

    # Test with mean pooling
    print("=== Similarity Search with Mean Pooling ===")
    print(f"Query: {query}\n")
    results_mean = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=True, top_k=3)
    for i, (text, score) in enumerate(results_mean, 1):
        print(f"{i}. Score: {score:.4f}\nText: {text}\n")

    # Test with CLS token (no mean pooling)
    print("=== Similarity Search with CLS Token ===")
    print(f"Query: {query}\n")
    results_cls = similarity_search(
        query, texts, model, tokenizer, use_mean_pooling=False, top_k=3)
    for i, (text, score) in enumerate(results_cls, 1):
        print(f"{i}. Score: {score:.4f}\nText: {text}\n")


if __name__ == "__main__":
    main()
