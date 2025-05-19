import json
import os
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file
from sklearn.utils import deprecation

# Suppress warnings (from prior responses)
# os.environ["OMP_NESTED"] = "FALSE"


def preprocess_texts(headers: List[dict]) -> List[str]:
    """
    Filter out noisy texts (e.g., menus, short texts) from headers.
    Args:
        headers: List of header dicts with 'text' key.
    Returns:
        List of cleaned texts.
    """
    exclude_keywords = ["menu", "sign in", "trending", "close"]
    min_words = 10
    return [
        header["text"]
        for header in headers
        if not any(keyword in header["text"].lower() for keyword in exclude_keywords)
        and len(header["text"].split()) >= min_words
    ]


def embed_search(
    query: str,
    texts: List[str],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20
) -> List[dict]:
    """
    Perform embedding-based search to retrieve top-k relevant texts.
    Args:
        query: Search query.
        texts: List of corpus texts.
        model_name: Sentence Transformer model.
        device: Device for encoding (mps for M1).
        top_k: Number of candidates to retrieve.
    Returns:
        List of dicts with text, score, and embedding.
    """
    model = SentenceTransformer(model_name, device=device)
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device)
    text_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    similarities = util.cos_sim(query_embedding, text_embeddings)[
        0].cpu().numpy()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "text": texts[i],
            "score": float(similarities[i]),
            "embedding": text_embeddings[i].cpu().numpy()
        }
        for i in top_k_indices
    ]


def rerank_results(
    query: str,
    candidates: List[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
) -> List[dict]:
    """
    Rerank candidates using a cross-encoder.
    Args:
        query: Search query.
        candidates: List of candidate dicts with 'text' and 'score'.
        model_name: Cross-encoder model.
        device: Device for encoding.
    Returns:
        Reranked list of dicts with updated scores.
    """
    model = CrossEncoder(model_name, device=device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def mmr_diversity(
    candidates: List[dict],
    num_results: int = 5,
    lambda_param: float = 0.5
) -> List[dict]:
    """
    Select diverse results using Maximal Marginal Relevance.
    Args:
        candidates: List of candidate dicts with 'text', 'rerank_score', 'embedding'.
        num_results: Number of final results.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
    Returns:
        List of diverse results.
    """
    selected = []
    candidate_embeddings = np.array([c["embedding"] for c in candidates])

    while len(selected) < num_results and candidates:
        if not selected:
            best_candidate = candidates.pop(0)
            selected.append(best_candidate)
        else:
            mmr_scores = []
            selected_embeddings = np.array([c["embedding"] for c in selected])
            for i, candidate in enumerate(candidates):
                relevance = candidate["rerank_score"]
                # Convert tensor to NumPy array after computing cosine similarity
                similarity = np.max(
                    util.cos_sim(
                        torch.tensor(candidate["embedding"]).to(
                            "mps" if torch.backends.mps.is_available() else "cpu"),
                        torch.tensor(selected_embeddings).to(
                            "mps" if torch.backends.mps.is_available() else "cpu")
                    )[0].cpu().numpy()
                )
                mmr_score = lambda_param * relevance - \
                    (1 - lambda_param) * similarity
                mmr_scores.append(mmr_score)

            best_idx = np.argmax(mmr_scores)
            selected.append(candidates.pop(best_idx))

    return selected


def search_diverse_context(
    query: str,
    headers: List[dict],
    model_name: str = "all-MiniLM-L12-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20,
    num_results: int = 5,
    lambda_param: float = 0.5
) -> List[dict]:
    """
    Search for diverse context data given a query.
    Args:
        query: Search query.
        headers: List of header dicts with 'text'.
        model_name: Sentence Transformer model.
        rerank_model: Cross-encoder model.
        device: Device for encoding.
        top_k: Number of candidates for reranking.
        num_results: Number of final diverse results.
        lambda_param: MMR relevance-diversity trade-off.
    Returns:
        List of dicts with text, score, rerank_score, and embedding.
    """
    texts = preprocess_texts(headers)
    if not texts:
        return []

    candidates = embed_search(query, texts, model_name, device, top_k)
    reranked = rerank_results(query, candidates, rerank_model, device)
    diverse_results = mmr_diversity(reranked, num_results, lambda_param)

    return diverse_results


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/headers.json"
    headers = load_file(docs_file)
    query = "best Isekai anime 2024"

    results = search_diverse_context(
        query=query,
        headers=headers,
        model_name="all-MiniLM-L12-v2",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        top_k=20,
        num_results=5,
        lambda_param=0.5
    )

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {json.dumps(result['text'])[:100]}...")
        print(f"Embedding Score: {result['score']:.4f}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print("-" * 50)

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    save_file(results, f"{output_dir}/search_with_mmr_results.json")
