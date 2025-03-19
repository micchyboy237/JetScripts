import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

# Load optimized model for Mac M1
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Faster than BGE
device = "mps" if torch.backends.mps.is_available(
) else "cpu"  # Use Metal if available

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME).to(device)
model.eval()  # Set to evaluation mode


def fast_rerank(query: str, documents: List[str], ids: List[str]) -> List[Dict]:
    """
    Optimized reranker for Mac M1 using a small cross-encoder model.

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
            score = outputs.logits.item()  # Extract ranking score

            scores.append({
                "id": doc_id,
                "score": score,
                "text": doc
            })

    # Sort by score in descending order
    return sorted(scores, key=lambda x: x["score"], reverse=True)
