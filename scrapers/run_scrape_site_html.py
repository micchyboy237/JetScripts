import json
from urllib.parse import urlparse

from jet.code.splitter_markdown_utils import count_md_header_contents, extract_md_header_contents, get_md_header_contents
from jet.data.utils import generate_unique_hash
from jet.scrapers.crawler.web_crawler import WebCrawler
from jet.utils.commands import copy_to_clipboard
from jet.utils.markdown import extract_json_block_content
from jet.utils.object import extract_null_keys
from jet.vectors.reranker.bm25_helpers import SearchResult, HybridSearch
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.batch_runner import BatchEvalRunner
from llama_index.core.evaluation.correctness import CorrectnessEvaluator
from llama_index.core.evaluation.faithfulness import FaithfulnessEvaluator
from jet.llm.evaluators.helpers.guideline_evaluator import CONTEXT_EVAL_GUIDELINES, GuidelineContextEvaluator, GuidelineEvaluator
from llama_index.core.evaluation.relevancy import RelevancyEvaluator
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.schema import Document
from tqdm import tqdm
import hrequests
from jet.actions.generation import call_ollama_chat
from jet.scrapers.browser.playwright import PageContent, scrape_sync, setup_sync_browser_page
from jet.scrapers.preprocessor import extract_header_contents, get_header_contents, html_to_markdown, scrape_markdown
from jet.search.scraper import scrape_url
from jet.search.searxng import search_searxng
from jet.utils.class_utils import class_to_string
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, Field, HttpUrl
from typing import Any, Generator, List, Optional, Type, TypedDict
from datetime import date
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

from jet.actions.vector_semantic_search import VectorSemanticSearch
from jet.logger import logger
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from jet.scrapers.utils import clean_text, extract_paragraphs, extract_text_elements
from jet.token.token_utils import filter_texts, get_model_max_tokens, get_ollama_tokenizer, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.file.utils import load_file, save_data, save_file
from jet.transformers.object import make_serializable
from jet.wordnet.similarity import filter_highest_similarity, search_similarities
from jet.llm.ollama.base import Ollama, VectorStoreIndex

# RANDOM_SEED = random.randint(0, 1000)
RANDOM_SEED = 42
LLM_OPTIONS = {
    "seed": RANDOM_SEED,
    "temperature": 0,
}

PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Query:
{query}
Answer:
"""

RELEVANCY_EVAL_TEMPLATE = PromptTemplate("""
Your task is to evaluate whether the response to the query aligns with the provided context information.
You have two options to answer: YES or NO.
Answer YES if the response to the query is in line with the context information; otherwise, answer NO.

Query and Response:
{query_str}

Context:
{context_str}

Answer:
""")

RELEVANCY_REFINE_TEMPLATE = PromptTemplate("""
We want to understand if the following query and response are in line with the context information:
{query_str}
We have provided an existing YES/NO answer:
{existing_answer}
We have the opportunity to refine the existing answer (only if needed) with additional context below.
------------
{context_msg}
------------
If the existing answer was already YES, still answer YES.
If the information is present in the new context, answer YES.
Otherwise, answer NO.
""")

RESPONSE_EVAL_GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    "The response should be comprehensive, ensuring all relevant information from the context is included and nothing essential is omitted.",
]


class NoResultsFoundError(Exception):
    """Custom exception to be raised when no results are found."""
    pass


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

    # Simulating the search function with the placeholder for your search logic
    results: list[SearchResult] = search_searxng(
        query_url="http://searxng.local:8080/search",
        query=query,
        min_score=2.0,
        filter_sites=filter_sites,
        engines=engines,
        config={
            "port": 3101
        },
    )

    if not results:
        raise NoResultsFoundError(f"No results found for query: '{query}'")

    return results


def scrape_urls(urls: list[str], output_dir: str = "generated") -> Generator[tuple[str, str], None, None]:

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

    includes_all = []
    excludes = []
    max_depth = 0

    crawler = WebCrawler(
        excludes=excludes, includes_all=includes_all, max_depth=max_depth)

    for start_url in urls:
        for result in crawler.crawl(start_url):
            yield start_url, result["html"]

    crawler.close()


class GuidelineEvalResult(EvaluationResult):
    guideline: str


def run_evaluate_response_guidelines(model: OLLAMA_MODEL_NAMES, query: str, contexts: list[str], response: str, guidelines: list[str] = RESPONSE_EVAL_GUIDELINES, output_cls: Optional[Type[BaseModel]] = None) -> list[GuidelineEvalResult]:
    llm = Ollama(model=model, temperature=0.75)
    output_parser = PydanticOutputParser(
        output_cls=output_cls) if output_cls else None

    evaluators = [
        GuidelineEvaluator(llm=llm, guidelines=guideline,
                           output_parser=output_parser)
        for guideline in guidelines
    ]

    results = []
    for guideline, evaluator in zip(guidelines, evaluators):
        eval_result = evaluator.evaluate(
            query=query,
            contexts=contexts,
            response=response,
        )
        print("=====")
        print(f"Guideline: {guideline}")
        print(f"Passing: {eval_result.passing}")
        print(f"Score: {eval_result.score}")
        print(f"Feedback: {eval_result.feedback}")
        results.append(GuidelineEvalResult(
            guideline=guideline, **eval_result.model_dump()))
    return results


if __name__ == "__main__":
    embed_model = "mxbai-embed-large"
    rerank_model = "all-minilm:33m"
    chat_model = "mistral"
    # chat_model = "gemma3:1b"
    eval_model = "gemma3:4b"
    chat_max_tokens = get_model_max_tokens(chat_model)

    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper"

    valid_id_topics = [
        "Philippines National ID Registration Tips 2025",
        "Philippines Passport Registration Tips 2025",
        "Philippines Driver’s License Registration Tips 2025",
        "Philippines SSS ID Registration Tips 2025",
        "Philippines GSIS eCard Registration Tips 2025",
        "Philippines PhilHealth ID Registration Tips 2025",
        "Philippines TIN ID Registration Tips 2025",
        "Philippines PRC ID Registration Tips 2025",
        "Philippines Voter’s ID Registration Tips 2025",
        "Philippines UMID Registration Tips 2025",
        "Philippines Postal ID Registration Tips 2025",
        "Philippines Barangay ID Registration Tips 2025",
        "Philippines Senior Citizen ID Registration Tips 2025",
        "Philippines PWD ID Registration Tips 2025",
        "Philippines OFW e-Card Registration Tips 2025",
        "Philippines Company ID Registration Tips 2025",
        "Philippines School ID Registration Tips 2025",
        "Philippines Indigenous Peoples ID Registration Tips 2025",
        "Philippines Solo Parent ID Registration Tips 2025",
        "Philippines Police Clearance and NBI Clearance Tips 2025"
    ]

    online_seller_topics = [
        "Philippines TikTok Seller Tips 2025",
        "Philippines Shopee Seller Tips 2025",
        "Philippines Lazada Seller Tips 2025",
        "Philippines Facebook Marketplace Seller Tips 2025",
        "Philippines Instagram Shop Seller Tips 2025",
        "Philippines Carousell Seller Tips 2025",
        "Philippines Zalora Seller Tips 2025",
        "Philippines Amazon Seller Tips 2025",
        "Philippines eBay Seller Tips 2025",
        "Philippines Shopify Seller Tips 2025",
        "Philippines Etsy Seller Tips 2025",
        "Philippines YouTube Shopping Seller Tips 2025",
        "Philippines Affiliate Marketing Seller Tips 2025",
        "Philippines Dropshipping Business Tips 2025",
        "Philippines Print-on-Demand Seller Tips 2025",
        "Philippines Digital Products Selling Tips 2025",
        "Philippines Wholesale and Reselling Tips 2025",
        "Philippines Online Food Selling Tips 2025",
        "Philippines Live Selling Strategies 2025",
        "Philippines Online Selling Legal Requirements 2025"
    ]

    all_topics = valid_id_topics + online_seller_topics

    chat_llm = Ollama(model=chat_model)
    eval_llm = Ollama(model=eval_model, temperature=0.75)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=eval_llm)
    relevancy_evaluator = RelevancyEvaluator(
        llm=eval_llm,
        eval_template=RELEVANCY_EVAL_TEMPLATE,
        refine_template=RELEVANCY_REFINE_TEMPLATE,
    )

    # correctness_evaluator = CorrectnessEvaluator(llm=eval_llm)
    # runner = BatchEvalRunner(
    #     {
    #         "faithfulness": faithfulness_evaluator,
    #         "relevancy": relevancy_evaluator,
    #         # "correctness": correctness_evaluator
    #     },
    #     workers=4,
    # )

    for topic in all_topics:
        sub_dir = f"{output_dir}/{topic.replace(" ", "_").lower()}"
        search_results = search_data(topic)

        urls = [item["url"] for item in search_results]

        # if os.path.isfile(doc_file):
        #     doc_texts = load_file(doc_file)
        # else:

        query = f"Given the context information, extract all texts relevant to the topic.\nOutput as a structured markdown with # headers.\nTopic: {topic}"
        top_search_n = 5

        doc_texts = scrape_urls(urls, output_dir=output_dir)

        largest_doc_headers_info = "", "", 0
        search_n_count = 0
        for url, html in doc_texts:
            if search_n_count >= top_search_n:
                break

            prev_html = largest_doc_headers_info[1]
            prev_header_count = largest_doc_headers_info[2]

            md_text = html_to_markdown(html)
            header_count = count_md_header_contents(md_text)
            if not prev_header_count or header_count >= prev_header_count:
                if header_count == prev_header_count:
                    prev_md_text = html_to_markdown(prev_html)
                    if len(md_text) > len(prev_md_text):
                        largest_doc_headers_info = url, html, header_count
                else:
                    largest_doc_headers_info = url, html, header_count

            search_n_count += 1

        # Save HTML
        html_file = f"{sub_dir}/scraped_html.html"
        url, html, _ = largest_doc_headers_info
        save_file(html, html_file)

        # Save extracted headers
        md_text = html_to_markdown(html)
        header_contents = get_md_header_contents(md_text)

        header_texts = [header["content"] for header in header_contents]
        header_text = "\n\n\n".join(header_texts)
        headers_file = f"{sub_dir}/headers.md"
        save_file(header_text, headers_file)

        header_contents = extract_md_header_contents(
            md_text, min_tokens_per_chunk=256, max_tokens_per_chunk=int(chat_max_tokens * 0.65), tokenizer=get_ollama_tokenizer(chat_model).encode)

        outputs = []
        eval_outputs: dict[str, EvaluationResult |
                           list[EvaluationResult]] = {}
        # passing_results = []
        for header_idx, header in enumerate(header_contents):
            context: str = header["content"]
            message = PROMPT_TEMPLATE.format(context=context, query=query)

            response = chat_llm.chat(message, options=LLM_OPTIONS)
            output: str = response.message.content
            output = output.strip()
            copy_to_clipboard(output)

            outputs.append(f"<!-- Answer {header_idx + 1} -->\n\n{output}")

            output_file = f"{sub_dir}/chat_results.md"
            save_file("\n\n\n".join(outputs), output_file)

            # Evaluate context relevancy to query

        #     # Evaluate results
        #     logger.newline()
        #     logger.info("Evaluate Results")

        #     eval_file = f"{sub_dir}/eval_results.json"

        #     # Faithfulness
        #     logger.newline()
        #     logger.orange("Evaluating faitfulness...")
        #     eval_result = faithfulness_evaluator.evaluate(
        #         query=query,
        #         contexts=[context],
        #         response=output,
        #     )
        #     passing_results.append(eval_result.passing)
        #     eval_outputs["faitfulness"] = eval_result
        #     save_file(eval_outputs, eval_file)

        #     # # Relevancy
        #     # logger.newline()
        #     # logger.orange("Evaluating relevancy...")
        #     # eval_result = relevancy_evaluator.evaluate(
        #     #     query=query,
        #     #     contexts=[context],
        #     #     response=output,
        #     # )
        #     # passing_results.append(eval_result.passing)
        #     # eval_outputs["relevancy"] = eval_result
        #     # save_file(eval_outputs, eval_file)

        #     # Guidelines
        #     logger.newline()
        #     logger.orange("Evaluating guidelines...")
        #     eval_results = run_evaluate_response_guidelines(
        #         model=eval_model,
        #         query=query,
        #         contexts=[context],
        #         response=output,
        #     )
        #     passing_results.extend(
        #         [eval_result.passing for eval_result in eval_results])
        #     eval_outputs["guidelines"] = eval_results
        #     save_file(eval_outputs, eval_file)

        # passed = all(passing_results)
        # if passed:
        #     logger.success("All eval results passed!")
        #     break
        # else:
        #     logger.warning("Some eval results failed!")
