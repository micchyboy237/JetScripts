from typing import Any, Union, List, Dict, Optional, Literal, TypedDict, DefaultDict
import numpy as np
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Fused similarity score.
        percent_difference: Percentage difference from the highest score.
        text: The compared text.
        relevance: Optional relevance score (e.g., from user feedback).
        word_count: Number of words in the text.
    """
    id: str
    rank: Optional[int]
    doc_index: int
    score: float
    percent_difference: Optional[float]
    text: str
    relevance: Optional[float]
    word_count: Optional[int]


def generate_key(text: str, query: str = None) -> str:
    """Generate a unique key for a text-query pair."""
    combined = (text + (query or "")).encode('utf-8')
    return hashlib.md5(combined).hexdigest()


@lru_cache(maxsize=1000)
def get_embedding_function(model_name: str) -> callable:
    """Load and cache a SentenceTransformer model."""
    logger.info(f"Loading model: {model_name}")
    return SentenceTransformer(model_name).encode


def preprocess_text(text: str, domain: Optional[str] = None) -> str:
    """
    Preprocess text for domain-specific applications.

    Args:
        text: Input text.
        domain: Optional domain (e.g., 'medical', 'legal') for specialized preprocessing.

    Returns:
        Preprocessed text.
    """
    # Example: Basic preprocessing (extend for domain-specific needs)
    text = text.lower().strip()
    if domain == "medical":
        # Placeholder for medical entity normalization
        pass
    return text


class VectorIndex:
    """Manages a FAISS index for efficient similarity search."""

    def __init__(self, dimension: int):
        # L2 for Euclidean; use IndexFlatIP for cosine
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.ids = []

    def add(self, embeddings: np.ndarray, texts: List[str], ids: List[str]):
        """Add embeddings to the index."""
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(ids)

    def search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """Search for top-k similar texts."""
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices


def query_similarity_scores(
    query: Union[str, List[str]],
    texts: Union[str, List[str]],
    threshold: float = 0.0,
    model: Union[str, List[str]] = "all-MiniLM-L6-v2",
    fuse_method: Literal["average", "max", "min", "weighted"] = "average",
    model_weights: Optional[List[float]] = None,
    ids: Union[List[str], None] = None,
    metrics: Literal["cosine", "dot", "euclidean"] = "cosine",
    domain: Optional[str] = None,
    use_index: bool = True,
    top_k: int = 100
) -> List[SimilarityResult]:
    """
    Computes similarity scores for queries against texts with scalable vector indexing and advanced fusion.

    Args:
        query: Single query or list of queries.
        texts: Single text or list of texts.
        threshold: Minimum similarity score to include.
        model: One or more embedding model names.
        fuse_method: Fusion method ('average', 'max', 'min', 'weighted').
        model_weights: Weights for weighted fusion (must match model count).
        ids: Optional list of IDs for texts.
        metrics: Similarity metric ('cosine', 'euclidean', 'dot').
        domain: Optional domain for preprocessing (e.g., 'medical').
        use_index: Whether to use FAISS indexing for large corpora.
        top_k: Number of top results to retrieve when using index.

    Returns:
        List of SimilarityResult, sorted by score with ranks and metadata.
    """
    # Input normalization
    query = [query] if isinstance(query, str) else query
    texts = [texts] if isinstance(texts, str) else texts
    model = [model] if isinstance(model, str) else model

    # Validation
    if not query or not texts:
        raise ValueError("Query and texts must be non-empty.")
    if not model:
        raise ValueError("At least one model must be provided.")
    if ids is not None and len(ids) != len(texts):
        raise ValueError(
            f"Length of ids ({len(ids)}) must match texts ({len(texts)}).")
    if fuse_method == "weighted" and (not model_weights or len(model_weights) != len(model)):
        raise ValueError(
            "Model weights must match number of models for weighted fusion.")
    if fuse_method not in {"average", "max", "min", "weighted"}:
        raise ValueError(f"Unsupported fusion method: {fuse_method}")
    if metrics not in {"cosine", "dot", "euclidean"}:
        raise ValueError(f"Unsupported metrics: {metrics}")

    text_ids = ids if ids else [generate_key(text, query[0]) for text in texts]
    preprocessed_texts = [preprocess_text(text, domain) for text in texts]
    preprocessed_queries = [preprocess_text(q, domain) for q in query]

    all_results = []

    # Initialize vector index if enabled
    vector_index = None
    if use_index:
        embed_func = get_embedding_function(model[0])
        sample_embedding = embed_func([preprocessed_texts[0]])[0]
        vector_index = VectorIndex(dimension=len(sample_embedding))
        text_embeddings = embed_func(preprocessed_texts)
        vector_index.add(text_embeddings, preprocessed_texts, text_ids)

    def process_model(model_name: str):
        results = []
        embed_func = get_embedding_function(model_name)

        if use_index:
            query_embeddings = embed_func(preprocessed_queries)
            for i, q_emb in enumerate(query_embeddings):
                distances, indices = vector_index.search(
                    q_emb[np.newaxis, :], top_k)
                scores = 1 / \
                    (1 + distances[0]
                     ) if metrics == "euclidean" else distances[0]
                for j, idx in enumerate(indices[0]):
                    if scores[j] >= threshold:
                        results.append({
                            "id": vector_index.ids[idx],
                            "doc_index": idx,
                            "query": query[i],
                            "text": vector_index.texts[idx],
                            "score": float(scores[j])
                        })
        else:
            query_embeddings = embed_func(preprocessed_queries)
            text_embeddings = embed_func(preprocessed_texts)

            if metrics == "cosine":
                query_norms = np.linalg.norm(
                    query_embeddings, axis=1, keepdims=True)
                text_norms = np.linalg.norm(
                    text_embeddings, axis=1, keepdims=True)
                query_embeddings /= np.where(query_norms != 0, query_norms, 1)
                text_embeddings /= np.where(text_norms != 0, text_norms, 1)
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "dot":
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "euclidean":
                similarity_matrix = np.zeros((len(query), len(texts)))
                for i in range(len(query)):
                    for j in range(len(texts)):
                        dist = np.linalg.norm(
                            query_embeddings[i] - text_embeddings[j])
                        similarity_matrix[i, j] = 1 / (1 + dist)

            for i, q in enumerate(query):
                scores = similarity_matrix[i]
                mask = scores >= threshold
                filtered_indices = np.arange(len(texts))[mask]
                filtered_scores = scores[mask]
                sorted_indices = np.argsort(filtered_scores)[::-1]
                for idx, j in enumerate(sorted_indices):
                    results.append({
                        "id": text_ids[filtered_indices[j]],
                        "doc_index": int(filtered_indices[j]),
                        "query": q,
                        "text": texts[filtered_indices[j]],
                        "score": float(filtered_scores[j])
                    })
        return results

    # Parallelize model processing
    with ThreadPoolExecutor() as executor:
        model_results = list(executor.map(process_model, model))
        for results in model_results:
            all_results.extend(results)

    # Fuse results
    fused_results = fuse_all_results(
        all_results,
        method=fuse_method,
        model_weights=model_weights if fuse_method == "weighted" else None
    )

    # Add metadata (word_count, relevance placeholder)
    for result in fused_results:
        result["word_count"] = len(result["text"].split())
        # Placeholder for external relevance signals
        result["relevance"] = None

    return fused_results


def fuse_all_results(
    results: List[Dict[str, Any]],
    method: str = "average",
    model_weights: Optional[List[float]] = None
) -> List[SimilarityResult]:
    """
    Fuses similarity results with advanced methods.

    Args:
        results: List of result dictionaries.
        method: Fusion method ('average', 'max', 'min', 'weighted').
        model_weights: Weights for weighted fusion.

    Returns:
        List of SimilarityResult, sorted by score.
    """
    # Step 1: Aggregate scores by (id, query, text)
    query_text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for result in results:
        key = (result["id"], result["query"], result["text"])
        query_text_data[key]["scores"].append(result["score"])
        query_text_data[key]["text"] = result["text"]
        query_text_data[key]["doc_index"] = result["doc_index"]

    # Step 2: Average scores across models
    query_text_averages = {
        key: {
            "text": data["text"],
            "score": float(sum(data["scores"]) / len(data["scores"])),
            "doc_index": data["doc_index"]
        }
        for key, data in query_text_data.items()
    }

    # Step 3: Fuse query-specific scores for each text
    text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for (id_, query, text), data in query_text_averages.items():
        text_key = (id_, text)
        text_data[text_key]["scores"].append(data["score"])
        text_data[text_key]["text"] = text
        text_data[text_key]["doc_index"] = data["doc_index"]

    # Step 4: Apply fusion method
    fused_scores = []
    for key, data in text_data.items():
        scores = data["scores"]
        if method == "average":
            score = float(sum(scores) / len(scores))
        elif method == "max":
            score = float(max(scores))
        elif method == "min":
            score = float(min(scores))
        elif method == "weighted":
            normalized_weights = [w / sum(model_weights)
                                  for w in model_weights]
            score = float(
                sum(s * w for s, w in zip(scores, normalized_weights[:len(scores)])))
        fused_scores.append({
            "id": key[0],
            "rank": None,
            "doc_index": data["doc_index"],
            "score": score,
            "percent_difference": None,
            "text": key[1]
        })

    # Step 5: Sort and assign ranks
    sorted_scores = sorted(
        fused_scores, key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(sorted_scores):
        result["rank"] = idx + 1

    # Step 6: Calculate percent_difference
    if sorted_scores:
        max_score = sorted_scores[0]["score"]
        for result in sorted_scores:
            result["percent_difference"] = round(
                abs(max_score - result["score"]) / max_score * 100, 2
            ) if max_score != 0 else 0.0

    return sorted_scores


def search_engine_rerank(
    queries: List[str],
    documents: List[str],
    document_ids: List[str],
    threshold: float = 0.3,
    top_k: int = 10
) -> List[SimilarityResult]:
    """
    Reranks search engine results for multiple queries using weighted model fusion.

    Args:
        queries: List of query variations (e.g., synonyms).
        documents: List of document texts to rerank.
        document_ids: Unique IDs for documents.
        threshold: Minimum similarity score to include.
        top_k: Number of top results to return.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Reranking {len(documents)} documents for {len(queries)} queries")
    results = query_similarity_scores(
        query=queries,
        texts=documents,
        ids=document_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2", "distilbert-base-nli-stsb-mean-tokens"],
        fuse_method="weighted",
        model_weights=[0.6, 0.4],  # Favor MiniLM for speed
        metrics="cosine",
        domain="general",
        use_index=len(documents) > 1000,  # Use FAISS for large corpora
        top_k=top_k
    )
    return results


def ecommerce_product_search(
    query: str,
    product_descriptions: List[str],
    product_ids: List[str],
    threshold: float = 0.4,
    top_k: int = 5
) -> List[SimilarityResult]:
    """
    Reranks e-commerce product listings based on a single query.

    Args:
        query: User's search query (e.g., "wireless headphones").
        product_descriptions: List of product description texts.
        product_ids: Unique IDs for products.
        threshold: Minimum similarity score to include.
        top_k: Number of top products to return.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Searching {len(product_descriptions)} products for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=product_descriptions,
        ids=product_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="ecommerce",  # Custom preprocessing for product descriptions
        use_index=len(product_descriptions) > 500,
        top_k=top_k
    )
    return results


def academic_paper_recommendation(
    query: str,
    paper_abstracts: List[str],
    paper_ids: List[str],
    citation_counts: List[int],
    threshold: float = 0.5,
    top_k: int = 10
) -> List[SimilarityResult]:
    """
    Recommends academic papers based on a research query, boosting scores by citation counts.

    Args:
        query: Research query (e.g., "deep learning for NLP").
        paper_abstracts: List of paper abstract texts.
        paper_ids: Unique IDs for papers.
        citation_counts: Citation counts for papers (used for relevance boosting).
        threshold: Minimum similarity score to include.
        top_k: Number of top papers to return.

    Returns:
        List of reranked SimilarityResult objects with relevance scores.
    """
    logger.info(f"Recommending papers for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=paper_abstracts,
        ids=paper_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2", "allenai-specter"],
        fuse_method="weighted",
        model_weights=[0.4, 0.6],  # Favor SPECTER for academic texts
        metrics="cosine",
        domain="academic",
        use_index=len(paper_abstracts) > 1000,
        top_k=top_k
    )

    # Boost scores based on citation counts
    max_citations = max(citation_counts) if citation_counts else 1
    for result in results:
        doc_index = result["doc_index"]
        citation_boost = citation_counts[doc_index] / \
            max_citations if max_citations > 0 else 0
        result["relevance"] = result["score"] * \
            (0.8 + 0.2 * citation_boost)  # Combine similarity and citations

    # Re-sort by relevance
    results.sort(key=lambda x: x["relevance"] or x["score"], reverse=True)
    for idx, result in enumerate(results):
        result["rank"] = idx + 1

    return results


def support_ticket_prioritization(
    urgency_queries: List[str],
    ticket_descriptions: List[str],
    ticket_ids: List[str],
    threshold: float = 0.5,
    top_k: int = 20
) -> List[SimilarityResult]:
    """
    Prioritizes customer support tickets based on urgency-related queries.

    Args:
        urgency_queries: Queries indicating urgency (e.g., "urgent issue", "critical error").
        ticket_descriptions: List of ticket description texts.
        ticket_ids: Unique IDs for tickets.
        threshold: Minimum similarity score to include.
        top_k: Number of top tickets to prioritize.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(f"Prioritizing {len(ticket_descriptions)} tickets")
    results = query_similarity_scores(
        query=urgency_queries,
        texts=ticket_descriptions,
        ids=ticket_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="max",  # Use max to capture any urgent signal
        metrics="cosine",
        domain="support",
        use_index=len(ticket_descriptions) > 1000,
        top_k=top_k
    )
    return results


def main():
    """
    Demonstrates real-world usage of semantic reranking for multiple scenarios.
    """
    # Example 1: Search Engine Reranking
    print("\n=== Search Engine Reranking ===")
    search_queries = ["best AI tools", "AI software recommendations"]
    search_docs = [
        "Top 10 AI tools for 2025 include TensorFlow and PyTorch.",
        "AI software is revolutionizing industries.",
        "Learn Python for AI development."
    ]
    search_ids = ["doc1", "doc2", "doc3"]
    search_results = search_engine_rerank(
        search_queries, search_docs, search_ids, top_k=3)
    for result in search_results:
        print(
            f"Rank {result['rank']}: {result['text']} (Score: {result['score']:.3f})")

    # Example 2: E-Commerce Product Search
    print("\n=== E-Commerce Product Search ===")
    product_query = "wireless headphones"
    product_descs = [
        "Bluetooth wireless headphones with noise cancellation.",
        "Wired earbuds with high-fidelity sound.",
        "Wireless speakers for home use."
    ]
    product_ids = ["prod1", "prod2", "prod3"]
    product_results = ecommerce_product_search(
        product_query, product_descs, product_ids, top_k=2)
    for result in product_results:
        print(
            f"Rank {result['rank']}: {result['text']} (Score: {result['score']:.3f})")

    # Example 3: Academic Paper Recommendation
    print("\n=== Academic Paper Recommendation ===")
    paper_query = "deep learning for NLP"
    paper_abstracts = [
        "This paper explores deep learning models for natural language processing.",
        "A survey of machine learning techniques.",
        "Robotics and AI: A new frontier."
    ]
    paper_ids = ["paper1", "paper2", "paper3"]
    citation_counts = [100, 50, 10]  # Example citation counts
    paper_results = academic_paper_recommendation(
        paper_query, paper_abstracts, paper_ids, citation_counts, top_k=2
    )
    for result in paper_results:
        print(
            f"Rank {result['rank']}: {result['text']} (Relevance: {result['relevance']:.3f})")

    # Example 4: Customer Support Ticket Prioritization
    print("\n=== Customer Support Ticket Prioritization ===")
    urgency_queries = ["urgent issue", "critical error"]
    ticket_descs = [
        "System crashed with critical error message.",
        "Slow performance on login page.",
        "Feature request for new dashboard."
    ]
    ticket_ids = ["ticket1", "ticket2", "ticket3"]
    ticket_results = support_ticket_prioritization(
        urgency_queries, ticket_descs, ticket_ids, top_k=2)
    for result in ticket_results:
        print(
            f"Rank {result['rank']}: {result['text']} (Score: {result['score']:.3f})")


if __name__ == "__main__":
    main()
