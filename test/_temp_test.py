from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class E5Embedder:
    def __init__(self, model_name="intfloat/e5-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts: list[str], is_query=False) -> torch.Tensor:
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + t for t in texts]
        encoded = self.tokenizer(
            prefixed_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded)
        embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / input_mask_expanded.sum(1)


if __name__ == "__main__":
    embedder = E5Embedder()

    # Sample passages and query
    passages = [
        "The Eiffel Tower is located in Paris.",
        "Python is a popular programming language.",
        "Transformers are used for NLP tasks."
    ]
    query = ["Where is the Eiffel Tower?"]

    # Embed
    passage_embeddings = embedder.embed(passages)
    query_embedding = embedder.embed(query, is_query=True)

    # Cosine similarity
    scores = torch.matmul(query_embedding, passage_embeddings.T)
    print("Similarity scores:", scores.tolist()[0])

    # Ranking
    top_idx = scores.squeeze().argsort(descending=True)
    print("Top results:")
    for idx in top_idx:
        print(f"{passages[idx]} - Score: {scores[0, idx].item():.4f}")
