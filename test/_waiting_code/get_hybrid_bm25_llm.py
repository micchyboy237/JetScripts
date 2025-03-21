from tqdm import tqdm
from jet.cache.cache_manager import CacheManager
from shared.data_types.job import JobData
from jet.wordnet.n_grams import count_ngrams
from jet.search.transformers import clean_string
from jet.logger import logger, time_it
from jet.file.utils import load_file
from jet.utils.commands import copy_to_clipboard
import torch
import math
from collections import Counter
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Plus


@time_it
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


def hybrid_search(queries: str | List[str], documents: List[str], ids: List[str], bm25_top_n=20) -> List[Dict]:
    """
    Hybrid search combining BM25+ for initial filtering and MiniLM for deep ranking.

    Args:
        queries (List[str]): Search queries.
        documents (List[str]): List of job descriptions.
        ids (List[str]): List of job IDs.
        bm25_top_n (int): Number of top BM25+ results to rerank.

    Returns:
        List[Dict]: Final ranked job applications.
    """
    if isinstance(queries, str):
        queries = []

    # Step 1: Use BM25+ to get top N results
    bm25_results = get_bm25p_with_auto_penalty(
        queries, documents, ids)[:bm25_top_n]

    if not bm25_results:  # No matches
        return []

    # Step 2: Extract top N documents
    filtered_docs = [entry["text"] for entry in bm25_results]
    filtered_ids = [entry["id"] for entry in bm25_results]

    # Step 3: Use MiniLM to rerank top N results
    query = " ".join(queries)
    final_results = fast_rerank(query, filtered_docs, filtered_ids)

    return final_results


@time_it
def prepare_inputs(queries: list[str]):
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    cache_dir = "generated/get_bm25_similarities"
    data: list[JobData] = load_file(data_file)

    cache_manager = CacheManager(cache_dir=cache_dir)

    # Load previous cache data
    cache_data = cache_manager.load_cache()

    if not cache_manager.is_cache_valid():
        sentences = []
        for item in data:
            sentence = "\n".join([
                item["title"],
                item["details"],
                "\n".join([f"Tech: {tech}" for tech in sorted(
                    item["entities"]["technology_stack"], key=str.lower)]),
                "\n".join([f"Tag: {tech}" for tech in sorted(
                    item["tags"], key=str.lower)]),
            ])
            cleaned_sentence = clean_string(sentence.lower())
            sentences.append(cleaned_sentence)

        # Generate n-grams
        common_texts_ngrams = [
            list(count_ngrams(sentence, max_words=5).keys()) for sentence in tqdm(sentences)
        ]

        # Update the cache with the new n-grams
        cache_data = cache_manager.update_cache(common_texts_ngrams)
    else:
        # Use the cached n-grams
        common_texts_ngrams = cache_data["common_texts_ngrams"]

    # Prepare queries and calculate BM25+ similarities
    query_ngrams = [list(count_ngrams(
        query, min_count=1, max_words=5)) for query in queries]
    data_dict = {item["id"]: item for item in data}
    ids = list(data_dict.keys())
    queries = ["_".join(text.split())
               for queries in query_ngrams for text in queries]

    common_texts = []
    for texts in common_texts_ngrams:
        formatted_texts = []
        for text in texts:
            formatted_texts.append("_".join(text.split()))
        common_texts.append(" ".join(formatted_texts))

    return data, {
        "queries": queries,
        "documents": common_texts,
        "ids": ids,
    }


if __name__ == "__main__":
    # Load optimized model for Mac M1
    # MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME).to(device)
    model.eval()

    queries = [
        "React.js",
        "React Native",
        "Web",
        "Mobile"
    ]

    data, inputs_dict = prepare_inputs(queries)
    data_dict = {item["id"]: item for item in data}

    similarities = hybrid_search(**inputs_dict)
    # Format the results
    results = [
        {
            "score": result["score"],
            # "similarity": result["similarity"],
            # "matched": result["matched"],
            "result": data_dict[result["id"]]
        }
        for result in similarities
    ]

    copy_to_clipboard({
        "count": len(results),
        "data": results[:50]
    })

    for idx, data in enumerate(results[:10]):
        result = data["result"]
        logger.log(f"{idx + 1}:", result["title"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{data["score"]:.2f}")
