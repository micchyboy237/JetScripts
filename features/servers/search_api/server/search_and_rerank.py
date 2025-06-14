from typing import AsyncIterator, Union
from collections import defaultdict
import json
import os
import shutil
from typing import AsyncIterator, Dict, List, Optional, Tuple, TypedDict, Iterator, Union
from datetime import datetime
import asyncio
from urllib.parse import unquote, urlparse
from jet.features.nltk_search import get_pos_tag, search_by_pos
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.llm.mlx.mlx_types import EmbedModelType, LLMModelType
from jet.llm.mlx.tasks.utils import load_model_components
from jet.logger import logger
from jet.scrapers.hrequests_utils import scrape_urls
from jet.transformers.link_formatters import LinkFormatter, format_links_for_embedding
from jet.utils.url_utils import rerank_bm25_plus
from jet.wordnet.text_chunker import truncate_texts
from jet.vectors.document_types import HeaderDocument
from jet.vectors.search_with_clustering import search_documents
from jet.wordnet.analyzers.text_analysis import ReadabilityResult, analyze_readability, analyze_text
from jet.code.splitter_markdown_utils import get_md_header_docs
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.models.tokenizer.base import count_tokens, get_tokenizer_fn
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls
from jet.scrapers.preprocessor import html_to_markdown
from jet.scrapers.utils import scrape_links, scrape_published_date, search_data
from jet.models.tasks.hybrid_search_docs_with_bm25 import search_docs
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance
from jet.wordnet.words import count_words
from jet.search.searxng import SearchResult

# Define the prompt template for LLM
PROMPT_TEMPLATE = """
You are a helpful assistant. Using the provided context, answer the following query as accurately and concisely as possible.

Query: {query}

Context:
{context}

Answer:
"""


class StepBackQueryResponse(TypedDict):
    original_query: str
    broader_query: List[str]


class ContextEntry(TypedDict):
    rank: int
    doc_index: int
    chunk_index: int
    tokens: int
    score: float
    rerank_score: float
    source_url: str
    parent_header: str
    header: str
    content: str


class ContextInfo(TypedDict):
    model: str
    total_tokens: int
    contexts: list[ContextEntry]


class StreamedStep(TypedDict):
    step_title: str
    step_result: Optional[Dict]


def get_header_stats(text: str) -> Dict:
    """Analyze text and return header statistics."""
    logger.debug("Analyzing text for header statistics")
    try:
        analysis = analyze_text(text)
        logger.info(
            f"Header stats computed: MTLD={analysis['mtld']}, Difficulty={analysis['overall_difficulty']}")
        return {
            "mtld": analysis["mtld"],
            "mtld_category": analysis["mtld_category"],
            "overall_difficulty": analysis["overall_difficulty"],
            "overall_difficulty_category": analysis["overall_difficulty_category"],
        }
    except Exception as e:
        logger.error(f"Error in get_header_stats: {str(e)}")
        return {}


async def filter_htmls_with_best_combined_mtld(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    limit: Optional[int] = None,
    min_mtld: float = 100.0
) -> List[Tuple[str, str, List[HeaderDocument]]]:
    logger.info(
        f"Filtering {len(url_html_date_tuples)} HTMLs with min MTLD={min_mtld}")
    doc_scores = []
    for url, html, _ in url_html_date_tuples:
        try:
            logger.debug(f"Processing HTML for URL: {url}")
            docs = get_md_header_docs(html, ignore_links=False)
            if len(docs) < 5:
                logger.warning(
                    f"Skipping {url}: insufficient headers ({len(docs)} < 5)")
                continue
            docs_text = "\n\n".join(doc.text for doc in docs)
            readability = analyze_readability(docs_text)
            mtld_score = readability["mtld"]
            if mtld_score >= min_mtld:
                doc_scores.append((url, html, docs, mtld_score))
                logger.debug(
                    f"Added {url} to candidates with MTLD={mtld_score}")
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            continue
    doc_scores.sort(key=lambda x: x[3], reverse=True)
    filtered = [(url, html, docs) for url, html, docs, _ in doc_scores[:limit]]
    logger.info(f"Filtered to {len(filtered)} HTMLs with highest MTLD scores")
    return filtered


def initialize_output_directory(script_path: str, sub_dir: Optional[str] = None) -> str:
    """Initialize output directory for storing results."""
    logger.debug(f"Initializing output directory for script: {script_path}")
    try:
        script_dir = os.path.dirname(os.path.abspath(script_path))
        base_dir = os.path.join(script_dir, "generated", os.path.splitext(
            os.path.basename(script_path))[0])
        output_dir = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory initialized: {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error initializing output directory: {str(e)}")
        raise


def initialize_search_components(
    llm_model: LLMModelType,
    embed_model: EmbedModelType,
    seed: int
) -> Tuple[MLX, callable]:
    """Initialize MLX model and tokenizer."""
    logger.debug(
        f"Initializing search components with LLM={llm_model}, Embed={embed_model}, Seed={seed}")
    try:
        mlx = MLX(llm_model, seed=seed)
        tokenize = get_tokenizer_fn(embed_model)
        logger.info("Search components initialized successfully")
        return mlx, tokenize
    except Exception as e:
        logger.error(f"Error initializing search components: {str(e)}")
        raise


async def fetch_search_results(query: str, output_dir: str, use_cache: bool = False) -> List[SearchResult]:
    """Fetch search results and save them."""
    logger.info(
        f"Fetching search results for query: {query}, use_cache={use_cache}")
    try:
        browser_search_results = search_data(query, use_cache=use_cache)
        logger.debug(f"Fetched {len(browser_search_results)} search results")
        save_file(
            {"query": query, "count": len(
                browser_search_results), "results": browser_search_results},
            os.path.join(output_dir, "browser_search_results.json")
        )
        return browser_search_results
    except Exception as e:
        logger.error(f"Error fetching search results: {str(e)}")
        return []


async def process_search_results(
    browser_search_results: List[dict],
    query: str,
    output_dir: str
) -> AsyncIterator[Union[StreamedStep, List[Tuple[str, str, Optional[str]]]]]:
    logger.info(
        f"Processing {len(browser_search_results)} search results for query: {query}")
    urls = [item["url"] for item in browser_search_results]

    # Process initial search result URLs
    html_list = []
    async for url, status, html in scrape_urls(urls, num_parallel=5):
        yield {
            "step_title": f"Scraping URL {status.capitalize()}",
            "step_result": {"url": url, "status": status}
        }
        if status == "completed":
            html_list.append(html)

    all_url_html_date_tuples = []
    all_links = []
    for result, html_str in zip(browser_search_results, html_list):
        url = result["url"]
        if not html_str:
            logger.debug(f"No HTML content for {url}, skipping")
            continue
        if not result.get("publishedDate"):
            published_date = scrape_published_date(html_str)
            result["publishedDate"] = published_date
            logger.debug(f"Scraped published date for {url}: {published_date}")
        links = set(scrape_links(html_str, url))
        links = [link for link in links if link != url]
        all_links.extend(links)
        all_url_html_date_tuples.append(
            (url, html_str, result.get("publishedDate")))

    all_links = list(set(all_links))
    save_file(all_links, os.path.join(output_dir, "links.json"))
    reranked_links = rerank_bm25_plus(all_links, query, 3)
    save_file(reranked_links, os.path.join(output_dir, "reranked_links.json"))

    # Process reranked links
    reranked_html_list = []
    async for url, status, html in scrape_urls(reranked_links, num_parallel=5):
        yield {
            "step_title": f"Scraping Reranked URL {status.capitalize()}",
            "step_result": {"url": url, "status": status}
        }
        if status == "completed":
            reranked_html_list.append(html)

    for url, html_str in zip(reranked_links, reranked_html_list):
        if html_str:
            published_date = scrape_published_date(html_str)
            all_url_html_date_tuples.append((url, html_str, published_date))

    logger.info(
        f"Processed {len(all_url_html_date_tuples)} URL-HTML-date tuples")
    yield all_url_html_date_tuples


async def process_documents(
    url_html_date_tuples: List[Tuple[str, str, Optional[str]]],
    output_dir: str,
    min_mtld: float
) -> List[HeaderDocument]:
    logger.info(f"Processing {len(url_html_date_tuples)} documents")
    all_url_docs_tuples = await filter_htmls_with_best_combined_mtld(url_html_date_tuples, min_mtld=min_mtld)
    all_docs = []
    headers = []
    for url, _, docs in all_url_docs_tuples:
        for doc in docs:
            doc.metadata["source_url"] = url
            headers.append({
                "doc_index": doc["doc_index"],
                "source_url": url,
                "parent_header": doc["parent_header"],
                "header": doc["header"],
            })
        all_docs.extend(docs)
    save_file(all_docs, os.path.join(output_dir, "docs.json"))
    save_file(headers, os.path.join(output_dir, "headers.json"))
    return all_docs


async def search_and_group_documents(
    query: str,
    all_docs: List[HeaderDocument],
    embed_model: str,
    llm_model: str,
    top_k: int,
    output_dir: str
) -> AsyncIterator[Union[StreamedStep, Tuple[List[dict], str, ContextInfo]]]:
    logger.info(f"Searching {len(all_docs)} documents for query: {query}")

    # Step 1: Filter documents
    yield {
        "step_title": "Filtering Documents",
        "step_result": {"message": f"Removing header_level=1 from {len(all_docs)} documents"}
    }
    docs_to_search = [
        doc for doc in all_docs if doc.metadata["header_level"] != 1]
    logger.debug(f"Filtered to {len(docs_to_search)} documents")
    yield {
        "step_title": "Documents Filtered",
        "step_result": {"count": len(docs_to_search)}
    }
    await asyncio.sleep(0)  # Yield control to event loop

    # Step 2: Search documents
    yield {
        "step_title": "Searching Documents",
        "step_result": {"message": f"Performing search with query: {query}, top_k={top_k}"}
    }
    search_doc_results = search_docs(
        query=query,
        documents=docs_to_search,
        ids=[doc.id_ for doc in docs_to_search],
        model=embed_model,
        top_k=top_k,
    )
    logger.debug(f"Search returned {len(search_doc_results)} results")
    save_file(
        {"query": query, "count": len(
            search_doc_results), "results": search_doc_results},
        os.path.join(output_dir, "search_doc_results.json")
    )
    yield {
        "step_title": "Document Search Completed",
        "step_result": {"count": len(search_doc_results)}
    }
    await asyncio.sleep(0)

    # Step 3: Sort results
    yield {
        "step_title": "Sorting Search Results",
        "step_result": {"message": "Sorting by source URL and doc index"}
    }
    sorted_doc_results = sorted(
        search_doc_results,
        key=lambda x: (x["document"]["metadata"]["source_url"], x["doc_index"])
    )
    save_file(
        {"query": query, "count": len(
            sorted_doc_results), "results": sorted_doc_results},
        os.path.join(output_dir, "sorted_doc_results.json")
    )
    logger.debug(f"Sorted {len(sorted_doc_results)} results")
    yield {
        "step_title": "Search Results Sorted",
        "step_result": {"count": len(sorted_doc_results)}
    }
    await asyncio.sleep(0)

    # Step 4: Count tokens
    logger.info(f"Counting contexts ({len(sorted_doc_results)}) tokens...")
    yield {
        "step_title": "Counting Tokens",
        "step_result": {"message": f"Counting tokens for {len(sorted_doc_results)} results"}
    }
    result_texts = [result["text"] for result in sorted_doc_results]
    context_tokens: List[int] = count_tokens(
        llm_model, result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)
    save_file(
        {
            "total_tokens": total_tokens,
            "contexts": [
                {
                    "doc_index": result["doc_index"],
                    "score": result["score"],
                    "tokens": tokens,
                    "text": result["text"]
                }
                for result, tokens in zip(sorted_doc_results, context_tokens)
            ]
        },
        os.path.join(output_dir, "contexts.json")
    )
    logger.info(
        f"Saved context with {context_tokens} tokens to {output_dir}/contexts.json")
    yield {
        "step_title": "Tokens Counted",
        "step_result": {"total_tokens": total_tokens}
    }
    await asyncio.sleep(0)

    # Step 5: Build context
    yield {
        "step_title": "Building Context",
        "step_result": {"message": "Grouping results by source URL"}
    }
    contexts = []
    context_info: ContextInfo = {
        "model": llm_model, "total_tokens": total_tokens, "contexts": []}
    current_url = None
    for idx, doc in enumerate(sorted_doc_results):
        source_url = doc["document"]["metadata"]["source_url"]
        if source_url != current_url:
            contexts.append(f"<!-- Source: {source_url} -->")
            current_url = source_url
        contexts.append(doc["text"])
        context_info["contexts"].append({
            "rank": doc["rank"],
            "doc_index": doc["doc_index"],
            "chunk_index": doc["document"]["metadata"].get("chunk_index", 0),
            "tokens": context_tokens[idx],
            "score": doc["score"],
            "rerank_score": doc.get("rerank_score", 0.0),
            "source_url": source_url,
            "parent_header": doc["document"]["metadata"]["parent_header"],
            "header": doc["document"]["metadata"]["header"],
            "content": doc["text"]
        })
    context = "\n\n".join(contexts)
    save_file(context, os.path.join(output_dir, "context.md"))
    logger.debug(f"Built context with {len(contexts)} segments")
    yield {
        "step_title": "Context Built",
        "step_result": {"context_segments": len(contexts)}
    }
    await asyncio.sleep(0)

    # Yield final result
    yield (sorted_doc_results, context, context_info)


async def generate_response(
    query: str,
    context: str,
    llm_model: str,
    mlx: MLX,
    output_dir: str
) -> Iterator[StreamedStep]:
    """Generate and stream LLM response with step titles."""
    logger.info(
        f"Generating response for query: {query} with model: {llm_model}")
    yield {
        "step_title": "Preparing Prompt",
        "step_result": {"message": "Constructing input for language model"}
    }
    try:
        prompt = PROMPT_TEMPLATE.format(query=query, context=context)
        logger.debug("Prompt prepared for LLM")
        yield {
            "step_title": "Generating Response",
            "step_result": {"message": "Starting response generation"}
        }
        response = ""
        for chunk in mlx.stream_chat(
            prompt,
            system_prompt=get_system_date_prompt(),
            temperature=0.7,
            verbose=False,
            max_tokens=10000
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            yield {
                "step_title": "Streaming Response Chunk",
                "step_result": {"content": content}
            }
            await asyncio.sleep(0)  # Yield control to event loop
        save_file(
            {"query": query, "context": context, "response": response},
            os.path.join(output_dir, "chat_response.json")
        )
        yield {
            "step_title": "Response Complete",
            "step_result": {
                "message": "Response generation finished",
                "full_response": response
            }
        }
        logger.info(f"Successfully generated response for query: {query}")
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield {
            "step_title": "Generation Error",
            "step_result": {"error": str(e)}
        }


async def evaluate_results(
    query: str,
    context: str,
    response: str,
    llm_model: str,
    output_dir: str
) -> Iterator[StreamedStep]:
    """Evaluate context and response relevance with streaming steps."""
    logger.info(f"Evaluating context relevance for query: {query}")
    yield {
        "step_title": "Initializing Evaluation",
        "step_result": {"message": "Preparing evaluation process"}
    }
    try:
        os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
        yield {
            "step_title": "Evaluating Context Relevance",
            "step_result": {"message": "Assessing context suitability"}
        }
        model_components = load_model_components(llm_model)
        eval_context_result = evaluate_context_relevance(
            query, context, model_components)
        save_file(
            eval_context_result,
            os.path.join(output_dir, "eval",
                         "evaluate_context_relevance_result.json")
        )
        yield {
            "step_title": "Context Evaluation Complete",
            "step_result": {"context_relevance": eval_context_result}
        }
        logger.info(f"Evaluating response relevance for query: {query}")
        yield {
            "step_title": "Evaluating Response Relevance",
            "step_result": {"message": "Assessing response quality"}
        }
        eval_response_result = evaluate_response_relevance(
            query, context, response, model_components)
        save_file(
            eval_response_result,
            os.path.join(output_dir, "eval",
                         "evaluate_response_relevance_result.json")
        )
        yield {
            "step_title": "Response Evaluation Complete",
            "step_result": {"response_relevance": eval_response_result}
        }
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        yield {
            "step_title": "Evaluation Error",
            "step_result": {"error": str(e)}
        }
        await asyncio.sleep(0)  # Yield control to event loop


async def main() -> Iterator[StreamedStep]:
    """Main function to orchestrate search with streaming feedback."""
    query = "List all ongoing and upcoming isekai anime 2025."
    top_k = 10
    embed_model = "static-retrieval-mrl-en-v1"
    llm_model = "llama-3.2-1b-instruct-4bit"
    seed = 45
    use_cache = False
    logger.info(f"Starting search engine with query: {query}")

    yield {
        "step_title": "Initializing Search",
        "step_result": {"message": "Setting up search parameters"}
    }

    output_dir = initialize_output_directory(__file__)
    yield {
        "step_title": "Output Directory Ready",
        "step_result": {"directory": output_dir}
    }

    try:
        mlx, _ = initialize_search_components(llm_model, embed_model, seed)
        yield {
            "step_title": "Search Components Initialized",
            "step_result": {"message": "Model and tokenizer loaded"}
        }

        browser_search_results = await fetch_search_results(query, output_dir, use_cache)
        yield {
            "step_title": "Search Results Fetched",
            "step_result": {"count": len(browser_search_results)}
        }

        url_html_date_tuples = await process_search_results(
            browser_search_results, query, output_dir
        )
        url_html_date_tuples.sort(key=lambda x: x[2] or "", reverse=True)
        yield {
            "step_title": "Search Results Processed",
            "step_result": {"count": len(url_html_date_tuples)}
        }

        all_docs = process_documents(url_html_date_tuples, output_dir)
        yield {
            "step_title": "Documents Processed",
            "step_result": {"count": len(all_docs)}
        }

        sorted_doc_results, context = search_and_group_documents(
            query, all_docs, embed_model, llm_model, top_k, output_dir
        )
        yield {
            "step_title": "Documents Searched and Grouped",
            "step_result": {"context_segments": len(context.split("\n\n"))}
        }

        response = ""
        for step in generate_response(query, context, llm_model, mlx, output_dir):
            if step["step_title"] == "Response Complete":
                response = step["step_result"]["full_response"]
            yield step

        for step in evaluate_results(query, context, response, llm_model, output_dir):
            yield step

        logger.info("Search engine execution completed")
        yield {
            "step_title": "Search Completed",
            "step_result": {"message": "Search process finished successfully"}
        }
    except Exception as e:
        logger.error(f"Search execution failed: {str(e)}")
        yield {
            "step_title": "Search Error",
            "step_result": {"error": str(e)}
        }

if __name__ == "__main__":
    logger.info("Starting search engine script")
    try:
        async def run_main():
            async for step in main():
                logger.debug(f"Main step: {step['step_title']}")
        asyncio.run(run_main())
        logger.info("Search engine script finished")
    except Exception as e:
        logger.error(f"Error running main: {str(e)}")
