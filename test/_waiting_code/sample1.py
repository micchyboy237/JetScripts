import json
import os
import random
from urllib.parse import urlparse

from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.data.utils import generate_unique_hash
from jet.llm.query.retrievers import query_llm_structured
from jet.scrapers.crawler.web_crawler import WebCrawler
from jet.utils.object import extract_null_keys
from jet.vectors.reranker.bm25_helpers import SearchResult, HybridSearch, SearchResultData, preprocess_texts
from jet.wordnet.stopwords import StopWords
from tqdm import tqdm
import hrequests
from jet.actions.generation import call_ollama_chat
from jet.scrapers.browser.playwright_helpers import PageContent, scrape_sync, setup_sync_browser_page
from jet.scrapers.preprocessor import extract_header_contents, get_header_contents, html_to_markdown, scrape_markdown
from jet.search.scraper import scrape_url
from jet.search.searxng import search_searxng
from jet.utils.class_utils import class_to_string
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, HttpUrl
from typing import Any, Generator, List, Optional, TypedDict
from datetime import date
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

from jet.actions.vector_semantic_search import VectorSemanticSearch
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.scrapers.utils import clean_text
from jet.token.token_utils import filter_texts, get_model_max_tokens, get_ollama_tokenizer, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_data, save_file
from jet.transformers.object import make_serializable
from jet.wordnet.similarity import filter_highest_similarity, search_similarities
from jet.llm.ollama.base import Ollama
from langchain_core.documents import Document

# RANDOM_SEED = random.randint(0, 1000)
RANDOM_SEED = 42


class Episode(BaseModel):
    season: int
    episode: int
    date: date
    title: Optional[str] = None
    summary: Optional[str] = None


class AnimeDetails(BaseModel):
    title: str
    seasons: Optional[int] = None
    status: Optional[str] = None
    synopsis: Optional[str] = None
    genre: Optional[List[str]] = None
    release_date: Optional[date] = None
    end_date: Optional[date] = None
    episodes: Optional[List[Episode]] = None


episode_fields = list(Episode.model_fields.keys())
anime_details_fields = list(AnimeDetails.model_fields.keys())

# class Anime(BaseModel):
#     id: int
#     title: str
#     synopsis: str
#     genre: List[str]
#     studio: Optional[str] = None
#     status: str  # Example: "Ongoing", "Completed", "Upcoming"
#     release_date: Optional[date] = None
#     end_date: Optional[date] = None
#     total_episodes: Optional[int] = None
#     seasons: List[Season] = []
#     poster_url: Optional[HttpUrl] = None
#     trailer_url: Optional[HttpUrl] = None
#     rating: Optional[float] = None  # Example: IMDb/MAL rating


class Anime(BaseModel):
    title: str
    seasons: int
    episodes: int
    synopsis: Optional[str] = None
    genre: Optional[List[str]] = None
    release_date: Optional[date] = None
    end_date: Optional[date] = None


# Get list of field names
anime_fields = list(Anime.model_fields.keys())


crawler = None


def setup_web_crawler(urls: list[str], includes: list[str] = [], excludes: list[str] = [], max_depth: int = 0):
    global crawler

    if not crawler:
        crawler = WebCrawler(urls=urls, excludes=excludes,
                             includes=includes, max_depth=max_depth)

    return crawler


def search_data(query) -> list[SearchResult]:
    include_sites = [
        # "https://easypc.com.ph",
        # "9anime",
        # "zoro"
        # "aniwatch"
        "myanimelist.net",
        "reelgood.com",
    ]
    exclude_sites = ["wikipedia.org"]
    engines = [
        "google",
        "brave",
        "duckduckgo",
        "bing",
        "yahoo",
    ]
    results: list[SearchResult] = search_searxng(
        query_url="http://searxng.local:8080/search",
        query=query,
        min_score=0.2,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
        engines=engines,
        config={
            "port": 3101
        },
    )
    # results_dict = {
    #     f"URL: {result["url"]}\n{result["title"]}\n{result["content"]}": result for result in results}
    # rerank_candidates = list(results_dict.keys())
    # reranked_results = search_similarities(
    #     query,
    #     candidates=rerank_candidates,
    #     model_name=rerank_model)
    # results = [results_dict[item["text"]] for item in reranked_results]
    return results


def html_extractor(html_str):
    markdown = html_to_markdown(html_str)
    header_contents = extract_header_contents(markdown)
    texts = [item["content"] for item in header_contents]
    return texts


def generate_browser_query(model: str, data: dict, *, seed: int = RANDOM_SEED) -> str:
    system = "You are an AI assistant that follows instructions. You read object keys and values to understand the provided data. You analyze all null values in the given data and identify missing information. You then generate a query to search on a browser to fill in the missing values. You ensure that the generated query is specific and relevant to the anime title provided. You provide a clear search query based on the gaps in the data for further research. You focus on completing the data by utilizing accurate and efficient search methods."

    prompt = f"Data:\n{json.dumps(data, indent=2)}"

    options = {
        "seed": seed,
        "temperature": 0.75,
    }

    response = ""
    for chunk in call_ollama_chat(prompt, model, system=system, options=options):
        response += chunk

    return response


def scrape_data(query: str, docs: list[Document], *, seed: int = RANDOM_SEED):
    texts = [clean_text(doc.page_content) for doc in docs]
    # LLM Query

    max_llm_tokens = 0.8
    contexts: list[str] = filter_texts(
        texts, llm_model, max_llm_tokens)
    context = "\n\n".join(contexts)

    llm = Ollama(model=llm_model)
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, schema and not prior knowledge, "
        "answer the query.\n"
        "The generated JSON must pass the provided schema when validated.\n"
        "Use null for unavailable values.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    response = llm.structured_predict(
        output_cls=output_cls,
        prompt=qa_prompt,
        context_str=context,
        query_str=query,
        llm_kwargs={
            "options": {
                "seed": seed,
                "temperature": 0
            },
            # "max_prediction_ratio": 0.5
        },
    )
    response_dict = make_serializable(response)
    logger.success(format_json(response_dict))
    return response_dict


def search_query_contents(search_results: list[SearchResult]):
    texts = [item["content"] for item in search_results]

    hybrid_search = HybridSearch()
    hybrid_search.build_index(texts)


class ScrapedUrlResult(TypedDict):
    url: str
    texts: list[str]
    data: SearchResultData
    is_complete: bool


def scrape_urls(urls: list[str], queries: str | list[str], output_dir: str = "generated", includes: list[str] = [], excludes: list[str] = []) -> Generator[ScrapedUrlResult, None, None]:

    if isinstance(queries, str):
        queries = [queries]

    queries = preprocess_texts(queries)

    # for url in tqdm(urls, desc="Scraping urls", unit="URL"):
    #     loader = RecursiveUrlLoader(
    #         url=url, max_depth=2
    #     )
    #     docs = loader.load()
    #     texts = [
    #         header_content
    #         for doc in docs
    #         for header_content in html_extractor(doc.page_content)
    #     ]
    #     doc_texts.extend(texts)

    max_depth = 0

    crawler = setup_web_crawler(
        urls=urls, includes=includes, excludes=excludes, max_depth=max_depth)
    hybrid_search = HybridSearch()

    scraped_results = {urlparse(url).hostname: [] for url in urls}
    all_results: list[SearchResult] = []
    all_texts: list[str] = []
    all_queries: list[str] = []

    search_results: Optional[SearchResultData] = None
    is_complete = False

    if is_complete:
        logger.success(f"Completed data for queries: {queries}")
        break

    for result in crawler.crawl():
        host_name = urlparse(result["url"]).hostname
        doc_texts = [
            header_content
            for header_content in html_extractor(result["html"])
        ]
        if len(doc_texts) < 2:
            continue

        scraped_results[host_name].extend(doc_texts)
        all_texts.extend(doc_texts)

        hybrid_search.build_index(all_texts)

        top_k = None
        threshold = 0.0
        search_results = hybrid_search.search(
            "\n".join(queries), top_k=top_k, threshold=threshold)

        all_results.extend(search_results["results"])
        all_queries.extend(
            [query for query in search_results["queries"] if query not in all_queries])

        # Aggregate all "matched"
        all_matched = {}
        for result in all_results:
            result_matched = result["matched"]
            for match_query, match in result_matched.items():
                if match_query not in all_matched:
                    all_matched[match_query] = 0
                all_matched[match_query] += 1

        if search_results["matched"]:
            is_complete = all(
                count for query, count in search_results["matched"].items()
                if query in queries
            )
            # is_complete = False
            yield {
                "url": start_url,
                "texts": doc_texts,
                "data": search_results,
                "is_complete": is_complete,
            }


def query_structured_data(query: str, top_k: Optional[int] = 10, output_dir: str = "generated", includes: list[str] = [], excludes: list[str] = []) -> Generator[ScrapedUrlResult, None, None]:
    search_results = search_data(query)

    urls = [item["url"] for item in search_results]

    # if os.path.isfile(doc_file):
    #     doc_texts = load_file(doc_file)
    # else:

    yield from scrape_urls(
        urls, query, output_dir=output_dir, includes=includes, excludes=excludes)

    # search = VectorSemanticSearch(
    #     candidates=doc_texts, embed_model=embed_model)

    # fusion_results = search.fusion_search(query)
    # embed_texts: list[str] = []
    # for query_idx, (query_line, group) in enumerate(fusion_results.items()):
    #     embed_texts.extend([g["text"] for g in group])
    # embed_texts = embed_texts[:top_k]
    # # logger.newline()
    # # logger.orange(f"Fusion Search Results ({len(reranked_header_contents)}):")

    # chunk_size = 200
    # chunk_overlap = 50
    # rerank_candidates = split_texts(
    #     embed_texts, rerank_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # reranked_texts = []

    # reranked_results = search_similarities(
    #     query, candidates=rerank_candidates, model_name=rerank_model)
    # reranked_texts.extend([item["text"] for item in reranked_results])

    # all_docs: list[Document] = []
    # for content in reranked_texts:
    #     all_docs.append(Document(page_content=content))

    # response_dict = scrape_data(query, all_docs)
    # return response_dict


def fill_null_values(data: dict, output_dir: str = "generated"):
    null_keys = extract_null_keys(data)
    if not null_keys:
        return data

    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in null_keys])
    query = f"Anime \"{title}\" {search_keys_str}"

    response_dict = query_structured_data(query, output_dir=output_dir)
    original_data = data.copy()

    def merge_dicts(original, updates):
        """Recursively merge updates into original only if original has null values."""
        for key, value in updates.items():
            if key in original:
                if isinstance(original[key], dict) and isinstance(value, dict):
                    merge_dicts(original[key], value)
                elif original[key] is None:
                    original[key] = value
        return original

    merged_result = merge_dicts(original_data, response_dict)
    return fill_null_values(merged_result, output_dir=output_dir)


if __name__ == "__main__":

    embed_model = "mxbai-embed-large"
    rerank_model = "all-minilm:33m"
    llm_model = "mistral"

    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data"

    title = "I'll Become a Villainess Who Goes Down in History"
    fields = anime_fields
    output_cls = AnimeDetails
    # output_cls = None
    top_k = None

    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in fields])
    scraped_results: dict[str, list[str]] = {}
    output_results: dict[str, SearchResultData] = {}
    query = f"Anime \"{title}\" {search_keys_str}"

    # stopwords = StopWords()
    # includes = [t for t in title.lower().split(
    # ) if t not in stopwords.english_stop_words]
    includes = [
        "myanimelist.net",
        "reelgood.com",
    ]
    excludes = [
        "wikipedia.org"
    ]

    search_results_gen = query_structured_data(
        query, top_k=top_k, output_dir=output_dir, includes=includes, excludes=excludes)

    base_output_dir = output_dir
    for idx, result in enumerate(search_results_gen):
        search_results = result["data"]
        is_complete = result["is_complete"]
        url = result["url"]

        output_dir = f"{base_output_dir}/{idx + 1}/{urlparse(url).hostname}"

        search_results = search_results.copy()
        semantic_results = search_results.pop("semantic_results")
        hybrid_results = search_results.pop("hybrid_results")
        reranked_results = search_results.pop("reranked_results")

        save_file({
            "url": url,
            **search_results,
        }, f"{output_dir}/results_info.json")
        save_file(result["texts"], f"{output_dir}/docs.json")
        save_file(semantic_results, f"{output_dir}/semantic_results.json")
        save_file(hybrid_results, f"{output_dir}/hybrid_results.json")
        save_file(reranked_results, f"{output_dir}/reranked_results.json")

        doc_file = f"{output_dir}/scraped_texts.json"
        search_result_texts = [result["text"]
                               for result in semantic_results]
        scraped_results[query] = search_result_texts
        save_file(scraped_results, doc_file)

        # all_results_file = f"{output_dir}/all_results.json"
        # save_file({
        #     "queries": all_queries,
        #     "matched": all_matched,
        #     "results": all_results,
        # }, all_results_file)

        output_file = f"{output_dir}/output.json"
        output_results[query] = search_results
        save_file(output_results, output_file)

        if is_complete:
            # Ask LLM
            system = None
            texts = [result["text"] for result in semantic_results]
            llm_response_stream = query_llm_structured(
                query, texts, model=llm_model, system=system, output_cls=output_cls)
            structued_llm_results = []
            for structued_llm_response in llm_response_stream:
                structued_llm_results.append(structued_llm_response)

                save_file(structued_llm_results,
                          f"{output_dir}/structued_llm_results.json")
                # break

    # response_dict = fill_null_values(response_dict)
    # save_file(response_dict, output_file)

    # keywords = [
    #     "title",
    #     "synopsis",
    #     "air_date",
    # ]
    # search_keys_str = ", ".join(
    #     [key.replace('.', ' ').replace('_', ' ') for key in keywords])
    # query = f"Anime \"{title}\" season 1 episodes 1-13 {search_keys_str}"
    # output_cls = AnimeDetails
    # response_dict = query_structured_data(query)
    # save_file(response_dict, output_file)
    # response_dict = fill_null_values(response_dict)
    # save_file(response_dict, output_file)

    if crawler:
        crawler.close()
