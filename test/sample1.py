import json
import random

from tqdm import tqdm
import hrequests
from jet.actions.generation import call_ollama_chat
from jet.scrapers.browser.playwright import PageContent, scrape_sync, setup_sync_browser_page
from jet.scrapers.preprocessor import extract_header_contents, get_header_contents, scrape_markdown
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
from jet.token.token_utils import filter_texts, get_model_max_tokens, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_file
from jet.transformers.object import make_serializable
from jet.wordnet.similarity import search_similarities
from jet.llm.ollama.base import Ollama
from langchain_core.documents import Document

RANDOM_SEED = random.randint(0, 1000)

# class Episode(BaseModel):
#     episode_number: int
#     title: str
#     synopsis: Optional[str] = None
#     air_date: Optional[date] = None
#     duration_minutes: Optional[int] = None
#     thumbnail_url: Optional[HttpUrl] = None


# class Season(BaseModel):
#     season_number: int
#     title: str
#     episodes: List[Episode]
#     release_date: Optional[date] = None
#     end_date: Optional[date] = None


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


output_cls = Anime


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
    results_dict = {
        f"URL: {result["url"]}\n{result["title"]}\n{result["content"]}": result for result in results}
    rerank_candidates = list(results_dict.keys())
    reranked_results = search_similarities(
        query, candidates=rerank_candidates, model_name=rerank_model)
    final_results = [results_dict[item["text"]] for item in reranked_results]
    return final_results


def html_extractor(html_str):
    markdown = scrape_markdown(html_str)
    header_contents = extract_header_contents(markdown["content"])
    texts = [item["content"] for item in header_contents]
    text = "\n\n".join(texts)
    return text


def scrape_urls(urls: list[str]) -> list[Document]:
    # all_docs: list[Document] = []
    # for url in urls:
    #     loader = RecursiveUrlLoader(
    #         url=url, max_depth=2, extractor=lambda x: html_extractor(x)
    #     )
    #     docs = loader.load()
    #     all_docs.extend(docs)
    # return all_docs

    rerank_candidates: list[str] = []
    for url in tqdm(urls, desc="Scraping urls", unit="URL"):
        html_str = scrape_url(url)
        markdown = scrape_markdown(html_str)
        header_contents = extract_header_contents(markdown["content"])

        if not header_contents:
            continue

        rerank_candidates.extend([result["content"]
                                 for result in header_contents])
        # splitted_rerank_candidates = split_texts(rerank_candidates, rerank_model)

    reranked_results = search_similarities(
        query, candidates=rerank_candidates, model_name=rerank_model)
    reranked_header_contents = [item["text"] for item in reranked_results]

    all_docs: list[Document] = []
    for content in reranked_header_contents:
        all_docs.append(Document(page_content=content))

    return all_docs


def generate_browser_query(model: str, data: dict, *, seed: int = RANDOM_SEED) -> str:
    system = "You are an AI assistant that follows instructions. You read object keys and values to understand the provided data. You analyze all null values in the given data and identify missing information. You then generate a query to search on a browser to fill in the missing values. You ensure that the generated query is specific and relevant to the anime title provided. You provide a clear search query based on the gaps in the data for further research. You focus on completing the data by utilizing accurate and efficient search methods.```"

    prompt = f"Data:\n{json.dumps(data, indent=2)}"

    options = {
        "seed": seed,
        "temperature": 0.3,
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
                "temperature": 0.3
            },
            # "max_prediction_ratio": 0.5
        },
    )
    response_dict = make_serializable(response)
    logger.success(format_json(response_dict))
    return response_dict


if __name__ == "__main__":

    embed_model = "mxbai-embed-large"
    rerank_model = "all-minilm:33m"
    llm_model = "llama3.2"

    output_file = "generated/search_web_data.json"

    title = "I'll Become a Villainess Who Goes Down in History"
    query = f"Anime: \"{title}\"\n\nSchema:\n{class_to_string(output_cls)}"

    search_results = search_data(query)
    search_results = search_results[:5]

    urls = [item["url"] for item in search_results]

    docs = scrape_urls(urls)
    response_dict = scrape_data(query, docs)

    save_file(response_dict, output_file)

    # Check remaining null values

    new_query = generate_browser_query(embed_model, response_dict)
    search_results = search_data(new_query)
    urls = [item["url"] for item in search_results]
    new_docs = scrape_urls(urls)
    new_response_dict = scrape_data(query, new_docs)

    save_file(new_response_dict, output_file)
