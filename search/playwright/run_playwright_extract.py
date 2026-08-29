import os
import shutil
from typing import List, Optional, TypedDict

from jet.adapters.llama_cpp.hybrid_utils import HybridSearchResult, hybrid_search
from jet.code.markdown_types import HeaderSearchResult
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.file.utils import save_file
from jet.logger import logger
from jet.scrapers.utils import search_data
from jet.search.playwright.playwright_extract import (
    PlaywrightExtract,
    convert_html_to_markdown,
)
from jet.utils.text import format_sub_dir

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)


class ContextItem(TypedDict):
    doc_idx: int
    tokens: int
    text: str


class Topic(TypedDict):
    rank: int
    doc_index: int
    score: float
    text: str


class SearchResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    tokens: int
    text: str


def sync_example(urls):
    """Demonstrate synchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = extractor._run(
            urls=urls,
            extract_depth="basic",
            include_images=False,
            include_favicon=False,
            format="markdown",
        )
        print("Basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result["results"]:
            print(f"- {item['url']} (Content length: {len(item['raw_content'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_1_result.json")
    except Exception as e:
        print(f"Error in basic extract: {e}")

    try:
        result = extractor._run(
            urls=urls,
            extract_depth="advanced",
            include_images=True,
            include_favicon=True,
            format="text",
        )
        print("\nAdvanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result["results"]:
            print(
                f"- {item['url']} (Images: {len(item['images'])}, Favicon: {item['favicon']})"
            )
        save_file(result, f"{OUTPUT_DIR}/example_2_result.json")
    except Exception as e:
        print(f"Error in advanced extract: {e}")


async def async_example(urls):
    """Demonstrate asynchronous usage of PlaywrightExtract."""
    extractor = PlaywrightExtract()
    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="basic",
            include_images=True,
            include_favicon=False,
            format="markdown",
        )
        print("\nAsync basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result["results"]:
            print(f"- {item['url']} (Images: {len(item['images'])})")
        print(f"Response time: {result['response_time']:.2f} seconds")
        save_file(result, f"{OUTPUT_DIR}/example_3_result.json")
    except Exception as e:
        print(f"Error in async basic extract: {e}")

    try:
        result = await extractor._arun(
            urls=urls,
            extract_depth="advanced",
            include_images=False,
            include_favicon=True,
            format="text",
        )
        print("\nAsync advanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result["results"]:
            print(f"- {item['url']} (Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_4_result.json")
    except Exception as e:
        print(f"Error in async advanced extract: {e}")


def extract_doc_chunks(
    html: str, url: str, chunk_size: int = 200, chunk_overlap: int = 50
) -> List[dict]:
    """Extract structured chunks from HTML content.

    Returns list of dicts with 'id' and 'content' keys, compatible with
    jet.adapters.llama_cpp.hybrid_utils.hybrid_search documents parameter.
    """
    md_content = convert_html_to_markdown(html, ignore_links=True)
    headings = derive_by_header_hierarchy(md_content, ignore_links=True)
    docs = [
        {"id": f"{url}#{i}", "content": f"{header['header']}\n{header['content']}"}
        for i, header in enumerate(headings)
        if header.get("content")
    ]
    logger.debug(f"extract_doc_chunks: extracted {len(docs)} chunks from {url}")
    return docs


def extract_topics(
    query: str, documents: List[str], top_k: Optional[int] = None
) -> List[Topic]:
    """Extract topics from documents using BERTopic adapter.

    Args:
        query: Search query to find relevant topics
        documents: List of documents to analyze
        top_k: Number of top topics to return (if None, return all)

    Returns:
        List of Topic objects with rank, doc_index, score, and text
    """
    if not documents:
        return []

    try:
        from jet.adapters.bertopic import BERTopicAdapter

        logger.info(
            f"Starting topic extraction for {len(documents)} documents via BERTopicAdapter"
        )

        adapter = BERTopicAdapter()
        results = adapter.find_relevant_topics(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        topics: List[Topic] = []
        for rank, r in enumerate(results, start=1):
            topics.append(
                {
                    "rank": rank,
                    "doc_index": r.get("doc_index", 0),
                    "score": float(r.get("score", 0.0)),
                    "text": r.get("text", ""),
                }
            )

        logger.info(f"Returning {len(topics)} topics")
        return topics

    except ImportError as e:
        logger.error(f"BERTopicAdapter not available: {e}")
        return _fallback_topic_extraction(query, documents, top_k)
    except Exception as e:
        logger.error(f"Error in topic extraction: {e}")
        return _fallback_topic_extraction(query, documents, top_k)


def _fallback_topic_extraction(
    query: str, documents: List[str], top_k: int = None
) -> List[Topic]:
    """Fallback topic extraction using simple keyword matching."""
    import re
    from collections import Counter

    logger.warning("Using fallback keyword-based topic extraction")
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    results = []

    for doc_idx, doc in enumerate(documents):
        doc_words = re.findall(r"\b\w+\b", doc.lower())
        word_counts = Counter(doc_words)
        common_words = query_words.intersection(set(doc_words))
        if common_words:
            score = sum(word_counts[word] for word in common_words) / len(doc_words)
            top_words = [word for word, _ in word_counts.most_common(5)]
            topic_text = " ".join(top_words)
            results.append(
                {
                    "rank": len(results) + 1,
                    "doc_index": doc_idx,
                    "score": float(score),
                    "text": topic_text,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]
    return results


def test_extract_topics():
    """Test the extract_topics function with sample data."""
    print("Testing extract_topics function...")
    test_documents = [
        "Machine learning algorithms are revolutionizing data analysis and pattern recognition in various industries.",
        "Deep learning neural networks require large datasets and significant computational power for training.",
        "Natural language processing enables computers to understand and generate human language effectively.",
        "Computer vision applications can identify and classify objects in images and videos with high accuracy.",
        "Data science combines statistical analysis, programming skills, and domain expertise to extract insights.",
        "Artificial intelligence is transforming healthcare, finance, and transportation sectors worldwide.",
        "Reinforcement learning agents learn optimal strategies through trial and error interactions with environments.",
        "Supervised learning algorithms use labeled training data to make accurate predictions on new examples.",
        "Unsupervised learning discovers hidden patterns in data without requiring labeled examples.",
        "Transfer learning allows models trained on one task to be adapted for related tasks efficiently.",
    ]
    test_queries = [
        "machine learning algorithms",
        "neural networks and deep learning",
        "data science and analytics",
        "artificial intelligence applications",
    ]

    for query in test_queries:
        print(f"\nTesting with query: '{query}'")
        try:
            topics = extract_topics(
                query=query,
                documents=test_documents,
            )
            print(f"Found {len(topics)} topics:")
            for topic in topics:
                print(f"  - {topic['text']} (Score: {topic['score']:.3f})")
        except Exception as e:
            print(f"Error: {e}")


def search_contexts(query: str, html: str, url: str) -> List[HeaderSearchResult]:
    """Search for relevant contexts using Hybrid Search (Vector + Rerank).

    Uses jet.adapters.llama_cpp.hybrid_utils.hybrid_search which provides:
    - Stage 1: Vector retrieval via llama_cpp embed adapter (batched, deduped)
    - Stage 2: Cross-encoder reranking via llama_cpp rerank adapter
    - Score normalization to 0-1 range

    Models are configured via jet.adapters.llama_cpp.config (EMBED_MODEL, RERANK_MODEL).
    """
    chunks = extract_doc_chunks(html, url)
    if not chunks:
        logger.warning(f"search_contexts: no chunks extracted from {url}")
        return []

    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    logger.info(
        f"Running hybrid search for '{query[:60]}...' on {len(texts)} chunks from {url}"
    )

    # Use hybrid_search: vector retrieval + cross-encoder reranking
    results: List[HybridSearchResult] = hybrid_search(
        query=query,
        documents=texts,
        top_n=5,
        vector_score_threshold=0.3,
        normalize_scores=True,
    )

    # Map HybridSearchResult back to HeaderSearchResult format
    search_results: List[HeaderSearchResult] = []
    for r in results:
        search_results.append(
            {
                "id": ids[r["index"]],
                "rank": r["rank"],
                "score": r["score"],
                "text": r["text"],
                "metadata": {
                    "vector_score": r["vector_score"],
                    "rerank_score_raw": r["rerank_score_raw"],
                },
            }
        )

    logger.info(f"search_contexts: returned {len(search_results)} results for {url}")
    return search_results


def scrape_urls_data(
    query: str,
    urls: List[str],
    use_cache: bool = True,
    url_limit: Optional[int] = None,
):
    """Scrape and process URLs.

    Embedding/rerank models are configured via jet.adapters.llama_cpp.config
    or environment variables (EMBED_MODEL, RERANK_MODEL). No model parameter needed.
    """
    sub_dir_query = format_sub_dir(query)
    base_output_dir = f"{OUTPUT_DIR}/{sub_dir_query}"
    shutil.rmtree(base_output_dir, ignore_errors=True)
    os.makedirs(base_output_dir, exist_ok=True)

    logger.info(
        f"scrape_urls_data: query='{query}', urls={len(urls)}, cache={use_cache}, limit={url_limit}"
    )

    if not urls:
        logger.info("No URLs provided, searching via search_data...")
        search_engine_results = search_data(query, use_cache=use_cache)
        urls = [r["url"] for r in search_engine_results]
        save_file(
            search_engine_results, f"{base_output_dir}/search_engine_results.json"
        )
        save_file(urls, f"{base_output_dir}/urls.json")
        logger.info(f"Found {len(urls)} URLs from search")

    extractor = PlaywrightExtract()
    result_stream = extractor._stream(
        urls=urls,
        extract_depth="advanced",
        include_images=True,
        include_favicon=True,
        format="text",
        url_limit=url_limit,
    )

    logger.info("Advanced extract results stream started:")
    count = 0
    all_headers = {}

    for result in result_stream:
        count += 1
        meta = result.copy().pop("meta")
        chunks = extract_doc_chunks(meta["html"], result["url"])
        documents = [chunk["content"] for chunk in chunks]

        # Search using hybrid adapter (no model param needed)
        search_results = search_contexts(query, meta["html"], result["url"])

        sub_dir_url = format_sub_dir(result["url"])
        logger.info(
            f"[{count}] URL: {sub_dir_url} (Images: {len(result.get('images', []))}, Favicon: {result.get('favicon')})"
        )

        save_file(
            {
                "query": query,
                "count": len(chunks),
                "chunks": chunks,
            },
            f"{base_output_dir}/{sub_dir_url}/chunks.json",
        )

        save_file(
            {
                "query": query,
                "count": len(documents),
                "documents": documents,
            },
            f"{base_output_dir}/{sub_dir_url}/documents.json",
        )

        md_content = convert_html_to_markdown(meta["html"], ignore_links=True)
        headers = derive_by_header_hierarchy(md_content, ignore_links=True)
        save_file(
            {
                "query": query,
                "count": len(headers),
                "headers": headers,
            },
            f"{base_output_dir}/{sub_dir_url}/headers.json",
        )
        all_headers[result["url"]] = headers

        save_file(result, f"{base_output_dir}/{sub_dir_url}/results.json")
        save_file(
            search_results, f"{base_output_dir}/{sub_dir_url}/search_results.json"
        )
        save_file(
            {
                "url": result["url"],
                "tokens": meta["tokens"],
            },
            f"{base_output_dir}/{sub_dir_url}/info.json",
        )
        save_file(meta["analysis"], f"{base_output_dir}/{sub_dir_url}/analysis.json")
        save_file(
            meta["text_links"], f"{base_output_dir}/{sub_dir_url}/text_links.json"
        )
        save_file(
            meta["image_links"], f"{base_output_dir}/{sub_dir_url}/image_links.json"
        )
        save_file(meta["html"], f"{base_output_dir}/{sub_dir_url}/page.html")
        save_file(meta["markdown"], f"{base_output_dir}/{sub_dir_url}/markdown.md")
        save_file(
            clean_markdown_links(meta["markdown"]),
            f"{base_output_dir}/{sub_dir_url}/markdown_no_links.md",
        )
        save_file(meta["md_tokens"], f"{base_output_dir}/{sub_dir_url}/md_tokens.json")
        save_file(meta["screenshot"], f"{base_output_dir}/{sub_dir_url}/screenshot.png")

    logger.info(f"scrape_urls_data: processed {count} URLs total")
    return all_headers


if __name__ == "__main__":
    urls = []
    query = "Top RAG context engineering tips 2026 reddit"
    url_limit = 20
    use_cache = True

    sub_dir_query = format_sub_dir(query)

    # No model parameter needed - adapters use config/env vars
    all_headers = scrape_urls_data(
        query, urls, use_cache=use_cache, url_limit=url_limit
    )
    save_file(all_headers, f"{OUTPUT_DIR}/{sub_dir_query}/all_headers.json")

    all_contexts = []
    for url, headers in all_headers.items():
        context = f"<!-- Source: {url} -->\n"
        combined_headers_string = "\n".join(
            [f"{header['header']}\n{header['content']}" for header in headers]
        )
        context += combined_headers_string
        all_contexts.append(context)

    all_contexts_str = "\n\n".join(all_contexts)
    save_file(all_contexts_str, f"{OUTPUT_DIR}/{sub_dir_query}/all_contexts.md")
    logger.info(f"Saved all_contexts.md ({len(all_contexts_str)} chars)")
