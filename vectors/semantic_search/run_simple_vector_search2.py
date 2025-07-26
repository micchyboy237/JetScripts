from typing import List, TypedDict
from sentence_transformers import SentenceTransformer, util
import torch


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name, device="mps")

    def search(self, query: str, corpus: List[str], top_k: int = 5) -> List[SearchResult]:
        # Encode the query and corpus
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)

        # Compute cosine similarities
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # Sort scores and map to results
        top_results = torch.topk(scores, k=min(top_k, len(corpus)))

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            results.append({
                "rank": rank,
                "score": score.item(),
                "text": corpus[idx]
            })

        return results


if __name__ == "__main__":
    corpus = [
        "The cat sat on the mat.",
        "Dogs are friendly animals.",
        "Cats purr when they are happy.",
        "Birds can fly high in the sky.",
        "The mat was sat on by the cat."
    ]

    query = "What do cats do when content?"

    search_engine = SemanticSearch()
    results = search_engine.search(query, corpus, top_k=3)

    for result in results:
        print(
            f"Rank: {result['rank']}, Score: {result['score']:.4f}, Text: {result['text']}")
