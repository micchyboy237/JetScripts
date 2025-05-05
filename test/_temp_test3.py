import json
from typing import Any, Callable, Union, List, Dict, Optional, Literal, TypedDict, DefaultDict
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
import numpy as np
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from bs4 import BeautifulSoup
import trafilatura
import re
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import torch.nn.functional as F


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Fused similarity score.
        percent_difference: Percentage difference from the highest score.
        text: The compared text (or chunk if long).
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
    model = SentenceTransformer(model_name)
    return lambda texts, use_mean_pooling=True: get_embeddings(texts, model, use_mean_pooling)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """
    Applies mean pooling to token embeddings, accounting for attention mask.

    Args:
        token_embeddings: Token embeddings from the model (batch_size, seq_len, hidden_size).
        attention_mask: Attention mask (batch_size, seq_len).

    Returns:
        Pooled embeddings (batch_size, hidden_size) as a NumPy array.
    """
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()


def get_embeddings(texts: List[str], model: SentenceTransformer, use_mean_pooling: bool = True) -> np.ndarray:
    """
    Computes embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts: List of input texts.
        model: SentenceTransformer model.
        use_mean_pooling: Whether to apply mean pooling (default: True).

    Returns:
        Embeddings as a NumPy array.
    """
    # Encode inputs using the SentenceTransformer
    encoded_input = model.encode(
        texts,
        convert_to_tensor=True,
        device=model.device,
        batch_size=32,
        show_progress_bar=False
    )

    if use_mean_pooling:
        # If you need custom mean pooling, you can implement it
        # Assuming encoded_input is already properly pooled
        return encoded_input.cpu().numpy()
    else:
        # Check the shape of encoded_input
        if encoded_input.dim() == 2:
            # Already a 2D tensor (batch_size, hidden_size), return as is
            return encoded_input.cpu().numpy()
        elif encoded_input.dim() == 3:
            # 3D tensor (batch_size, seq_len, hidden_size), extract CLS token
            return encoded_input[:, 0, :].cpu().numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {encoded_input.shape}")


def preprocess_text(
    text: str,
    domain: Optional[str] = None,
    chunk_size: int = 150,
    overlap: int = 50,
    split_fn: Callable[[str], List[str]] = sent_tokenize
) -> List[str]:
    """
    Preprocesses web-scraped text, returning chunks for long texts.

    Args:
        text: Raw web-scraped text.
        domain: Optional domain (e.g., 'news', 'ecommerce') for specialized preprocessing.
        chunk_size: Maximum token length per chunk (approximate).
        overlap: Number of tokens to overlap between chunks.
        split_fn: Function to split text into logical units (default: NLTK sentence tokenizer).

    Returns:
        List of preprocessed text chunks.
    """
    extracted = None
    try:
        extracted = trafilatura.extract(
            text,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
    except Exception as e:
        logger.warning(f"Trafilatura failed to extract content: {e}")

    if not extracted:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            for element in soup(['nav', 'footer', 'script', 'style']):
                element.decompose()
            extracted = soup.get_text(separator=' ', strip=True)
            logger.info("Fallback to BeautifulSoup successful")
        except Exception as e:
            logger.warning(
                f"BeautifulSoup fallback failed: {e}, using raw text")
            extracted = text

    text = re.sub(r'\s+', ' ', extracted.strip())
    text = re.sub(r'(click here|read more|sign up|log in|subscribe now)',
                  '', text, flags=re.IGNORECASE)
    text = text.lower()

    if domain == "news":
        text = re.sub(
            r'published on \d{4}-\d{2}-\d{2}|by [a-zA-Z\s]+', '', text)
    elif domain == "ecommerce":
        text = re.sub(r'\$\d+\.\d{2}|\d+% off', '', text)
    elif domain == "forum":
        text = re.sub(r'posted by [a-zA-Z0-9\s]+|re: ', '', text)

    words = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words
            current_length = len(overlap_words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    if not chunks or len(words) < chunk_size:
        chunks = []
        units = split_fn(text)
        current_chunk = []
        current_length = 0

        for unit in units:
            unit_words = word_tokenize(unit)
            if current_length + len(unit_words) <= chunk_size:
                current_chunk.append(unit)
                current_length += len(unit_words)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                overlap_words = word_tokenize(
                    ' '.join(current_chunk))[-overlap:] if overlap > 0 else []
                current_chunk = [
                    ' '.join(overlap_words)] if overlap_words else []
                current_chunk.append(unit)
                current_length = len(overlap_words) + len(unit_words)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

    return chunks if chunks else [""]


class VectorIndex:
    """Manages a FAISS index for efficient similarity search."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP()
        self.texts = []
        self.ids = []

    def add(self, embeddings: np.ndarray, texts: List[str], ids: List[str]):
        """Add embeddings to the index."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms != 0, norms, 1)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(ids)

    def search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """Search for top-k similar texts."""
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / np.where(norm != 0, norm, 1)
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
    top_k: int = 100,
    use_bm25: bool = False,
    bm25_k: int = 1000,
    use_mean_pooling: bool = False
) -> List[SimilarityResult]:
    """
    Computes similarity scores for queries against texts with scalable vector indexing and hybrid search.

    Args:
        query: Single query or list of queries.
        texts: Single text or list of texts.
        threshold: Minimum similarity score to include.
        model: One or more embedding model names.
        fuse_method: Fusion method ('average', 'max', 'min', 'weighted').
        model_weights: Weights for weighted fusion.
        ids: Optional list of IDs for texts.
        metrics: Similarity metric ('cosine', 'euclidean', 'dot').
        domain: Optional domain for preprocessing.
        use_index: Whether to use FAISS indexing.
        top_k: Number of top results to retrieve.
        use_bm25: Whether to use BM25 for initial candidate selection.
        bm25_k: Number of BM25 candidates to retrieve.
        use_mean_pooling: Whether to apply mean pooling to embeddings (default: False).

    Returns:
        List of SimilarityResult, sorted by score with ranks and metadata.
    """
    query = [query] if isinstance(query, str) else query
    texts = [texts] if isinstance(texts, str) else texts
    model = [model] if isinstance(model, str) else model

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
    preprocessed_queries = [preprocess_text(q, domain)[0] for q in query]

    flat_texts = []
    flat_ids = []
    doc_indices = []
    for doc_idx, chunks in enumerate(preprocessed_texts):
        for chunk in chunks:
            flat_texts.append(chunk)
            flat_ids.append(text_ids[doc_idx])
            doc_indices.append(doc_idx)

    candidate_texts = flat_texts
    candidate_ids = flat_ids
    candidate_doc_indices = doc_indices
    if use_bm25 and flat_texts:
        tokenized_texts = [text.split() for text in flat_texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query[0].split())
        top_indices = np.argsort(bm25_scores)[
            ::-1][:min(bm25_k, len(flat_texts))]
        candidate_texts = [flat_texts[i] for i in top_indices]
        candidate_ids = [flat_ids[i] for i in top_indices]
        candidate_doc_indices = [doc_indices[i] for i in top_indices]

    all_results = []

    vector_index = None
    if use_index and candidate_texts:
        embed_func = get_embedding_function(model[0])
        sample_embedding = embed_func(
            [candidate_texts[0]], use_mean_pooling=use_mean_pooling)[0]
        vector_index = VectorIndex(dimension=len(sample_embedding))
        text_embeddings = embed_func(
            candidate_texts, use_mean_pooling=use_mean_pooling)
        vector_index.add(text_embeddings, candidate_texts, candidate_ids)

    def process_model(model_name: str):
        results = []
        embed_func = get_embedding_function(model_name)

        if use_index and vector_index:
            query_embeddings = embed_func(
                preprocessed_queries, use_mean_pooling=use_mean_pooling)
            for i, q_emb in enumerate(query_embeddings):
                distances, indices = vector_index.search(
                    q_emb[np.newaxis, :], min(top_k, len(candidate_texts)))
                scores = distances[0] if metrics == "cosine" else 1 / \
                    (1 + distances[0])
                for j, idx in enumerate(indices[0]):
                    if scores[j] >= threshold:
                        results.append({
                            "id": vector_index.ids[idx],
                            "doc_index": candidate_doc_indices[idx],
                            "query": query[i],
                            "text": vector_index.texts[idx],
                            "score": float(scores[j])
                        })
        else:
            query_embeddings = embed_func(
                preprocessed_queries, use_mean_pooling=use_mean_pooling)
            text_embeddings = embed_func(
                candidate_texts, use_mean_pooling=use_mean_pooling)

            if metrics == "cosine":
                query_norms = np.linalg.norm(
                    query_embeddings, axis=1, keepdims=True)
                text_norms = np.linalg.norm(
                    text_embeddings, axis=1, keepdims=True)
                query_embeddings = query_embeddings / \
                    np.where(query_norms != 0, query_norms, 1)
                text_embeddings = text_embeddings / \
                    np.where(text_norms != 0, text_norms, 1)
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "dot":
                similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
            elif metrics == "euclidean":
                similarity_matrix = np.zeros(
                    (len(query), len(candidate_texts)))
                for i in range(len(query)):
                    for j in range(len(candidate_texts)):
                        dist = np.linalg.norm(
                            query_embeddings[i] - text_embeddings[j])
                        similarity_matrix[i, j] = 1 / (1 + dist)

            for i, q in enumerate(query):
                scores = similarity_matrix[i]
                valid_indices = np.where(scores >= threshold)[0]
                sorted_indices = valid_indices[np.argsort(
                    scores[valid_indices])[::-1]]
                for j in sorted_indices[:min(top_k, len(valid_indices))]:
                    results.append({
                        "id": candidate_ids[j],
                        "doc_index": candidate_doc_indices[j],
                        "query": q,
                        "text": candidate_texts[j],
                        "score": float(scores[j])
                    })
        return results

    with ThreadPoolExecutor() as executor:
        model_results = list(executor.map(process_model, model))
        for results in model_results:
            all_results.extend(results)

    fused_results = fuse_all_results(
        all_results,
        method=fuse_method,
        model_weights=model_weights if fuse_method == "weighted" else None
    )

    doc_results = defaultdict(
        lambda: {"scores": [], "text": "", "doc_index": None})
    for result in fused_results:
        doc_idx = result["doc_index"]
        doc_results[doc_idx]["scores"].append(result["score"])
        doc_results[doc_idx]["text"] = texts[doc_idx]
        doc_results[doc_idx]["doc_index"] = doc_idx

    final_results = []
    for doc_idx, data in doc_results.items():
        score = sum(data["scores"]) / \
            len(data["scores"]) if data["scores"] else 0.0
        final_results.append({
            "id": text_ids[doc_idx],
            "rank": None,
            "doc_index": doc_idx,
            "score": score,
            "percent_difference": None,
            "text": data["text"],
            "relevance": None,
            "word_count": len(word_tokenize(data["text"]))
        })

    final_results.sort(key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(final_results):
        result["rank"] = idx + 1

    if final_results:
        max_score = final_results[0]["score"]
        for result in final_results:
            result["percent_difference"] = round(
                abs(max_score - result["score"]) / max_score * 100, 2
            ) if max_score != 0 else 0.0

    return final_results


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
    query_text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for result in results:
        key = (result["id"], result["query"], result["text"])
        query_text_data[key]["scores"].append(result["score"])
        query_text_data[key]["text"] = result["text"]
        query_text_data[key]["doc_index"] = result["doc_index"]

    query_text_averages = {
        key: {
            "text": data["text"],
            "score": float(sum(data["scores"]) / len(data["scores"])),
            "doc_index": data["doc_index"]
        }
        for key, data in query_text_data.items()
    }

    text_data = defaultdict(
        lambda: {"scores": [], "text": None, "doc_index": None})
    for (id_, query, text), data in query_text_averages.items():
        text_key = (id_, text)
        text_data[text_key]["scores"].append(data["score"])
        text_data[text_key]["text"] = text
        text_data[text_key]["doc_index"] = data["doc_index"]

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
            "scores": scores,
            "score": score,
            "percent_difference": None,
            "text": key[1]
        })

    return fused_scores


def news_article_search(
    query: str,
    articles: List[str],
    article_ids: List[str],
    threshold: float = 0.4,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Reranks web-scraped news articles based on a query.

    Args:
        query: User query (e.g., "AI in healthcare").
        articles: List of scraped article texts.
        article_ids: Unique IDs for articles.
        threshold: Minimum similarity score.
        top_k: Number of top articles to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(f"Reranking {len(articles)} news articles for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=articles,
        ids=article_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="news",
        use_index=len(articles) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def tech_blog_recommendation(
    queries: List[str],
    posts: List[str],
    post_ids: List[str],
    threshold: float = 0.3,
    top_k: int = 5,
    use_bm25: bool = False
) -> List[SimilarityResult]:
    """
    Recommends web-scraped tech blog posts based on multiple interest queries.

    Args:
        queries: List of user interest queries (e.g., ["machine learning", "AI trends"]).
        posts: List of scraped blog post texts.
        post_ids: Unique IDs for posts.
        threshold: Minimum similarity score.
        top_k: Number of top posts to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Recommending {len(posts)} blog posts for {len(queries)} queries")
    results = query_similarity_scores(
        query=queries,
        texts=posts,
        ids=post_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2", "distilbert-base-nli-stsb-mean-tokens"],
        fuse_method="weighted",
        model_weights=[0.6, 0.4],
        metrics="cosine",
        domain="blog",
        use_index=len(posts) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def ecommerce_product_scraping(
    query: str,
    product_listings: List[str],
    listing_ids: List[str],
    threshold: float = 0.5,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Reranks scraped e-commerce product listings based on a query.

    Args:
        query: Product search query (e.g., "wireless earbuds").
        product_listings: List of scraped product listing texts.
        listing_ids: Unique IDs for listings.
        threshold: Minimum similarity score.
        top_k: Number of top listings to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Reranking {len(product_listings)} product listings for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=product_listings,
        ids=listing_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="ecommerce",
        use_index=len(product_listings) > 500,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def forum_thread_prioritization(
    query: str,
    threads: List[str],
    thread_ids: List[str],
    threshold: float = 0.4,
    top_k: int = 5,
    use_bm25: bool = True
) -> List[SimilarityResult]:
    """
    Prioritizes web-scraped forum threads based on a query.

    Args:
        query: Technical query (e.g., "Python error handling").
        threads: List of scraped forum thread texts.
        thread_ids: Unique IDs for threads.
        threshold: Minimum similarity score.
        top_k: Number of top threads to return.
        use_bm25: Whether to use BM25 for candidate selection.

    Returns:
        List of reranked SimilarityResult objects.
    """
    logger.info(
        f"Prioritizing {len(threads)} forum threads for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=threads,
        ids=thread_ids,
        threshold=threshold,
        model=["all-MiniLM-L6-v2"],
        fuse_method="average",
        metrics="cosine",
        domain="forum",
        use_index=len(threads) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100
    )
    return results


def main():
    """
    Demonstrates semantic reranking for diverse, long, and unstructured web-scraped content.
    """
    # Example 1: News Article Search
    logger.orange("\n=== Example 1: News Article Search ===")
    news_query = "AI in healthcare"
    news_articles = [
        """
        <h1>AI Revolution in Healthcare</h1>
        <p>Published on 2025-02-01 by Jane Smith</p>
        <p>Artificial intelligence is transforming healthcare with predictive diagnostics and personalized treatments. Machine learning models analyze patient data to predict diseases like cancer with high accuracy. Hospitals are adopting AI to streamline operations. Click here to read more about AI trends.</p>
        <p>Subscribe to our newsletter for daily updates!</p>
        """,
        """
        <h2>Tech Stocks Surge in 2025</h2>
        <p>Published on 2025-01-15 by John Doe</p>
        <p>Tech companies, including those in AI, are seeing record stock gains. However, healthcare remains a challenging sector for investors. Read more about market trends.</p>
        <div>Follow us on social media!</div>
        """,
        """
        <h1>New Gadgets for 2025</h1>
        <p>From smartwatches to AI-powered home devices, new gadgets are hitting the market. Learn about the latest tech innovations. Sign up for our tech newsletter!</p>
        """
    ]
    news_ids = ["news1", "news2", "news3"]
    news_results = news_article_search(
        news_query, news_articles, news_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(news_query)
    for result in news_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 2: Tech Blog Recommendation
    logger.orange("\n=== Example 2: Tech Blog Recommendation ===")
    blog_queries = ["machine learning advancements", "AI trends 2025"]
    blog_posts = [
        """
        <div class="post">
        <h2>The Future of Machine Learning</h2>
        <p>Machine learning advancements in 2025 include larger language models and better interpretability. Neural networks are now more efficient, thanks to sparsity techniques. This blog explores how ML is reshaping industries like finance and healthcare.</p>
        <p>Join our webinar to learn more! Sign up today.</p>
        <p>We also cover AI ethics, a critical topic for 2025. From bias mitigation to transparency, ethical AI is a priority.</p>
        </div>
        """,
        """
        <h1>AI Trends to Watch</h1>
        <p>Generative AI is booming, with applications in content creation and design. In 2025, expect AI to integrate with IoT for smarter homes. Click here to subscribe to our tech updates.</p>
        <p>Other trends include AI in autonomous vehicles and robotics.</p>
        <div>Follow our blog for more insights!</div>
        """,
        """
        <h2>Python for Beginners</h2>
        <p>Learn Python programming with our step-by-step guide. While not directly about AI, Python is the backbone of many ML frameworks.</p>
        <p>Sign up for our coding bootcamp!</p>
        """
    ]
    blog_ids = ["blog1", "blog2", "blog3"]
    blog_results = tech_blog_recommendation(
        blog_queries, blog_posts, blog_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(format_json(blog_queries))
    for result in blog_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 3: E-Commerce Product Scraping
    logger.orange("\n=== Example 3: E-Commerce Product Scraping ===")
    product_query = "wireless earbuds"
    product_listings = [
        """
        <div class="product">
        <h2>AirPods Pro 2 - $249.99</h2>
        <p>Experience immersive sound with the AirPods Pro 2. Features include active noise cancellation, spatial audio, and up to 6 hours of battery life. Perfect for music lovers and professionals. Add to cart now!</p>
        <p>20% off this week only!</p>
        <p>Compatible with iOS and Android devices. Includes wireless charging case.</p>
        </div>
        """,
        """
        <h2>Sony WF-1000XM5 - $299.99</h2>
        <p>The Sony WF-1000XM5 wireless earbuds offer industry-leading noise cancellation and high-resolution audio. With 8 hours of battery life and a compact design, they’re ideal for travel. Buy now and save 10%!</p>
        <p>Free shipping on orders over $50.</p>
        """,
        """
        <h2>LED Desk Lamp - $29.99</h2>
        <p>Brighten your workspace with this energy-efficient LED desk lamp. Adjustable brightness and color temperature. Not wireless earbuds, but a great deal! Click here to purchase.</p>
        """
    ]
    listing_ids = ["prod1", "prod2", "prod3"]
    product_results = ecommerce_product_scraping(
        product_query, product_listings, listing_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(product_query)
    for result in product_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()

    # Example 4: Forum Thread Prioritization
    logger.orange("\n=== Example 4: Forum Thread Prioritization ===")
    forum_query = "Python error handling"
    forum_threads = [
        """
        <div class="thread">
        <p>Posted by User123: I'm getting a TypeError in Python when handling exceptions. Here's my code: <code>try: x = int(input()) except: print("Error")</code>. Any tips on proper error handling?</p>
        <p>Posted by DevGuru: You should specify the exception type, like <code>except ValueError:</code>. This avoids catching unrelated errors. Also, log the error with <code>logging</code>.</p>
        <p>Posted by User123: Thanks! That fixed it. Any libraries for advanced error handling?</p>
        </div>
        """,
        """
        <div class="thread">
        <p>Posted by CodeMaster: How do I optimize Python loops? My script is slow when processing large datasets.</p>
        <p>Posted by DataNerd: Use NumPy for vectorized operations or list comprehensions. Avoid nested loops where possible.</p>
        <p>Re: Check out the <code>multiprocessing</code> module for parallel processing.</p>
        </div>
        """,
        """
        <div class="thread">
        <p>Posted by Newbie: What's the best Python IDE? I'm new to coding.</p>
        <p>Posted by ProCoder: Try VS Code with Python extensions or PyCharm for advanced features.</p>
        </div>
        """
    ]
    thread_ids = ["thread1", "thread2", "thread3"]
    thread_results = forum_thread_prioritization(
        forum_query, forum_threads, thread_ids, top_k=2)
    logger.gray("Query:")
    logger.debug(forum_query)
    for result in thread_results:
        logger.log(
            f"Rank {result['rank']}:",
            f"(Score: {result['score']:.3f}, Words: {result['word_count']})",
            f"{result['text'][:100]}...",
            colors=["DEBUG", "SUCCESS", "WHITE"],
        )
        logger.newline()


def main2():
    import os
    import shutil

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    query = [
        "List upcoming isekai anime this year (2024-2025).",
    ]
    threshold = 0.0
    top_k = 10
    use_bm25 = True

    data: list[dict] = load_file(data_file)
    texts = [item["content"] for item in data]

    logger.info(
        f"Reranking {len(texts)} web scraped contents for query: {query}")
    results = query_similarity_scores(
        query=query,
        texts=texts,
        # ids=article_ids,
        threshold=threshold,
        model=["all-MiniLM-L12-v2", "distilbert-base-nli-stsb-mean-tokens"],
        fuse_method="average",
        metrics="cosine",
        domain=None,
        use_index=len(texts) > 1000,
        top_k=top_k,
        use_bm25=use_bm25,
        bm25_k=100,
        use_mean_pooling=True
    )

    logger.gray("Query:")
    logger.debug(query)
    for result in results[:5]:
        logger.log(
            f"Rank {result['rank']}:",
            f"Doc: {result['doc_index']}, Words: {result['word_count']}",
            f"\nScore: {result['score']:.3f}",
            f"\n{json.dumps(result['text'])[:100]}...",
            colors=["ORANGE", "DEBUG", "SUCCESS", "WHITE"],
        )

    save_file({
        "query": query,
        "results": results
    }, f"{output_dir}/query_scores.json")


if __name__ == "__main__":
    # main()
    main2()
