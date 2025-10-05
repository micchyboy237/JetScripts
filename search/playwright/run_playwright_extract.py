import os
import shutil
from jet.code.markdown_types import HeaderSearchResult
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.utils.text import format_sub_dir
from jet.vectors.semantic_search.header_vector_search import search_headers
import numpy as np
from typing import List, TypedDict
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy
from jet.search.playwright import PlaywrightExtract
from jet.file.utils import save_file
from jet.search.playwright.playwright_extract import convert_html_to_markdown

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

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
    rank: int
    doc_index: int
    score: float
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
            format="markdown"
        )
        print("Basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
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
            format="text"
        )
        print("\nAdvanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Images: {len(item['images'])}, Favicon: {item['favicon']})")
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
            format="markdown"
        )
        print("\nAsync basic extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
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
            format="text"
        )
        print("\nAsync advanced extract results:")
        print(f"Found {len(result['results'])} pages:")
        for item in result['results']:
            print(f"- {item['url']} (Favicon: {item['favicon']})")
        save_file(result, f"{OUTPUT_DIR}/example_4_result.json")
    except Exception as e:
        print(f"Error in async advanced extract: {e}")

def search_contexts(query: str, html: str, url: str, model: str) -> List[HeaderSearchResult]:
    md_content = convert_html_to_markdown(html, ignore_links=False)
    original_docs = derive_by_header_hierarchy(md_content, ignore_links=True)
    top_k = None
    threshold = 0.0
    chunk_size = 128
    chunk_overlap = 64
    search_results = list(
        search_headers(
            original_docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer_model=model,
            # merge_chunks=merge_chunks
        )
    )
    # contexts: List[ContextItem] = []
    # for idx, doc in enumerate(original_docs):
    #     doc["source"] = url
    #     text = doc["header"] + "\n\n" + doc["content"]
    #     contexts.append({
    #         "doc_idx": idx,
    #         "tokens": token_counter(text, model),
    #         "text": text,
    #     })
    return search_results

def extract_topics(
    query: str,
    documents: List[str],
    model: str = "nomic-embed-text-v2-moe",
    top_k: int = None
):
    pass

def search(
    query: str,
    documents: List[str],
    model: str = "nomic-embed-text-v2-moe",
    top_k: int = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.

    If top_k is None, return all results sorted by similarity.
    """
    if not documents:
        return []
    client = LlamacppEmbedding(model=model)
    vectors = client.get_embeddings([query] + documents, batch_size=1, show_progress=True)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    sorted_indices = np.argsort(similarities)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    return [
        {
            "rank": i + 1,
            "doc_index": int(sorted_indices[i]),
            "score": float(similarities[sorted_indices[i]]),
            "text": documents[sorted_indices[i]],
        }
        for i in range(len(sorted_indices))
    ]

def scrape_urls_data(query: str, urls: List[str], model: str):
    extractor = PlaywrightExtract()
    result_stream = extractor._stream(
        urls=urls,
        extract_depth="advanced",
        include_images=True,
        include_favicon=True,
        format="text"
    )
    print("\nAdvanced extract results stream:")
    count = 0
    # all_contexts = []
    # results = []
    for result in result_stream:
        count += 1
        meta = result.copy().pop("meta")
        search_results = search_contexts(query, meta["html"], result['url'], model)
        sub_dir_url = format_sub_dir(result['url'])
        print(f"URL: {sub_dir_url} (Images: {len(result['images'])}, Favicon: {result['favicon']})")
        # results.extend(result)
        # all_contexts.extend(contexts)
        save_file(result, f"{OUTPUT_DIR}/{sub_dir_url}/results.json")
        save_file(search_results, f"{OUTPUT_DIR}/{sub_dir_url}/search_results.json")
        save_file({
            "tokens": meta["tokens"],
        }, f"{OUTPUT_DIR}/{sub_dir_url}/info.json")
        save_file(meta["analysis"], f"{OUTPUT_DIR}/{sub_dir_url}/analysis.json")
        save_file(meta["text_links"], f"{OUTPUT_DIR}/{sub_dir_url}/text_links.json")
        save_file(meta["image_links"], f"{OUTPUT_DIR}/{sub_dir_url}/image_links.json")
        save_file(meta["html"], f"{OUTPUT_DIR}/{sub_dir_url}/page.html")
        save_file(meta["markdown"], f"{OUTPUT_DIR}/{sub_dir_url}/markdown.md")
        save_file(meta["md_tokens"], f"{OUTPUT_DIR}/{sub_dir_url}/md_tokens.json")
        save_file(meta["screenshot"], f"{OUTPUT_DIR}/{sub_dir_url}/screenshot.png")
    # return all_contexts

if __name__ == "__main__":
    urls = [
        "https://docs.tavily.com/documentation/api-reference/endpoint/crawl",
    ]
    model = "nomic-embed-text-v2-moe"
    query = "How to change max depth?"

    print("Running stream examples...")
    all_contexts = scrape_urls_data(query, urls, model)
    # save_file(all_contexts, f"{OUTPUT_DIR}/all_contexts.json")

    
    # texts = [doc["text"] for doc in all_contexts]
    # search_results = search(query, texts, model)
    # save_file({
    #     "query": query,
    #     "count": len(search_results),
    #     "results": search_results,
    # }, f"{OUTPUT_DIR}/search_results.json")

    # print("Running synchronous examples...")
    # sync_example(urls)
    # print("\nRunning asynchronous examples...")
    # asyncio.run(async_example(urls))
