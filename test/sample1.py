import json
import os
import random

from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.utils.object import extract_null_keys
from tqdm import tqdm
import hrequests
from jet.actions.generation import call_ollama_chat
from jet.scrapers.browser.playwright import PageContent, scrape_sync, setup_sync_browser_page
from jet.scrapers.preprocessor import extract_header_contents, get_header_contents, html_to_markdown, scrape_markdown
from jet.search.scraper import scrape_url
from jet.search.searxng import SearchResult, search_searxng
from jet.utils.class_utils import class_to_string
from llama_index.core.prompts.base import PromptTemplate
from pydantic.main import BaseModel
from pydantic import BaseModel, HttpUrl
from typing import Any, List, Optional
from datetime import date
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

from jet.actions.vector_semantic_search import VectorSemanticSearch
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.scrapers.utils import clean_text
from jet.token.token_utils import filter_texts, get_model_max_tokens, get_ollama_tokenizer, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_file
from jet.transformers.object import make_serializable
from jet.wordnet.similarity import filter_highest_similarity, search_similarities
from jet.llm.ollama.base import Ollama
from langchain_core.documents import Document

# RANDOM_SEED = random.randint(0, 1000)
RANDOM_SEED = 42


class Episode(BaseModel):
    episode_number: int
    title: str
    synopsis: Optional[str] = None
    air_date: Optional[date] = None
    duration_minutes: Optional[int] = None
    thumbnail_url: Optional[HttpUrl] = None


class Season(BaseModel):
    season_number: int
    title: str
    episodes: List[Episode]
    release_date: Optional[date] = None
    end_date: Optional[date] = None


class AnimeDetails(BaseModel):
    title: str
    seasons: List[Season] = []

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


keywords = [
    "seasons",
    "episodes",
    "synopsis",
    "genre",
    "release_date",
    "end_date",
]


def search_data(query) -> list[SearchResult]:
    filter_sites = [
        # "https://easypc.com.ph",
        # "9anime",
        # "zoro"
        # "aniwatch"
    ]
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
        filter_sites=filter_sites,
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


def scrape_urls(urls: list[str]) -> list[str]:
    docs_texts: list[str] = []

    all_docs: list[Document] = []
    for url in tqdm(urls, desc="Scraping urls", unit="URL"):
        loader = RecursiveUrlLoader(
            url=url, max_depth=2
        )
        docs = loader.load()
        texts = [
            header_content
            for doc in docs
            for header_content in html_extractor(doc.page_content)
        ]
        docs_texts.extend(texts)

    # chunk_overlap = 100
    # rerank_candidates = split_texts(
    #     docs_texts, embed_model, chunk_overlap=chunk_overlap)
    # reranked_results = search_similarities(
    #     query, candidates=docs_texts, model_name=embed_model)
    # docs_texts = [item["text"] for item in reranked_results]

    return docs_texts


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
        "Use null for empty values.\n"
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


def query_structured_data(query: str, top_k: int = 10):
    search_results = search_data(query)
    search_results = search_results[:5]

    urls = [item["url"] for item in search_results]

    doc_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    if os.path.isfile(doc_file):
        doc_texts = load_file(doc_file)
    else:
        doc_texts = scrape_urls(urls)
        save_file(doc_texts, doc_file)

    search = VectorSemanticSearch(
        candidates=doc_texts, embed_model=embed_model)

    fusion_results = search.fusion_search(query)
    embed_texts: list[str] = []
    for query_idx, (query_line, group) in enumerate(fusion_results.items()):
        embed_texts.extend([g["text"] for g in group])
    embed_texts = embed_texts[:top_k]
    # logger.newline()
    # logger.orange(f"Fusion Search Results ({len(reranked_header_contents)}):")

    chunk_overlap = 100
    rerank_candidates = split_texts(
        embed_texts, rerank_model, chunk_overlap=chunk_overlap)
    reranked_texts = []

    reranked_results = search_similarities(
        query, candidates=rerank_candidates, model_name=rerank_model)
    reranked_texts.extend([item["text"] for item in reranked_results])

    all_docs: list[Document] = []
    for content in reranked_texts:
        all_docs.append(Document(page_content=content))

    response_dict = scrape_data(query, all_docs)
    return response_dict


def fill_null_values(data: dict):
    null_keys = extract_null_keys(data)
    if not null_keys:
        return data

    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in null_keys])
    query = f"\"{data.get('title', '')}\" anime {search_keys_str}"

    response_dict = query_structured_data(query)
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
    return fill_null_values(merged_result)


if __name__ == "__main__":

    embed_model = "mxbai-embed-large"
    rerank_model = "all-minilm:33m"
    llm_model = "llama3.2"

    output_file = "generated/search_web_data.json"

    title = "I'll Become a Villainess Who Goes Down in History"

    search_keys_str = ", ".join(
        [key.replace('.', ' ').replace('_', ' ') for key in keywords])
    query = f"Anime \"{title}\" seasons and episodes"

    output_cls = Anime
    response_dict = query_structured_data(query)
    save_file(response_dict, output_file)
    # Check remaining null values
    response_dict = fill_null_values(response_dict)
    save_file(response_dict, output_file)

    output_cls = AnimeDetails
    response_dict = query_structured_data(query)
    save_file(response_dict, output_file)
    # Check remaining null values
    response_dict = fill_null_values(response_dict)
    save_file(response_dict, output_file)

    # new_query = generate_browser_query(embed_model, response_dict)
    # search_results = search_data(new_query)
    # urls = [item["url"] for item in search_results]
    # new_docs = scrape_urls(urls)
    # new_response_dict = scrape_data(query, new_docs)

    # save_file(new_response_dict, output_file)
