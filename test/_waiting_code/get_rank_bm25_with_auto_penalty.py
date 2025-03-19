import torch
import math
from collections import Counter
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Plus


def get_bm25p_with_auto_penalty(queries: List[str], documents: List[str], ids: List[str], k1=1.2, b=0.75, delta=1.0) -> List[Dict]:
    """
    Compute BM25+ similarities with an automatic penalty for irrelevant terms.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (List[str]): List of document ids.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor.

    Returns:
        List[Dict]: Ranked documents with normalized scores.
    """

    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / len(tokenized_docs)

    # Compute BM25+
    bm25 = BM25Plus(tokenized_docs, k1=k1, b=b, delta=delta)
    scores = bm25.get_scores(queries[0].lower().split())

    results = []
    for idx, score in enumerate(scores):
        if score > 0:  # Only return relevant results
            results.append({
                "id": ids[idx],
                "score": score,
                "text": documents[idx]
            })

    # Normalize scores
    if results:
        max_score = max(entry["score"] for entry in results)
        for entry in results:
            entry["score"] /= max_score if max_score > 0 else 1

    return sorted(results, key=lambda x: x["score"], reverse=True)


def fast_rerank(query: str, documents: List[str], ids: List[str]) -> List[Dict]:
    """
    Neural Reranker optimized for Mac M1 using MiniLM.

    Args:
        query (str): Search query.
        documents (List[str]): List of job descriptions.
        ids (List[str]): List of job IDs.

    Returns:
        List[Dict]: Sorted list of job applications ranked by relevance.
    """
    scores = []

    with torch.no_grad():  # No gradients needed for inference
        for doc, doc_id in zip(documents, ids):
            inputs = tokenizer(
                f"Query: {query} Document: {doc}", return_tensors="pt", truncation=True).to(device)
            outputs = model(**inputs)
            score = outputs.logits.item()

            scores.append({
                "id": doc_id,
                "score": score,
                "text": doc
            })

    return sorted(scores, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    # Load optimized model for Mac M1
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME).to(device)
    model.eval()
