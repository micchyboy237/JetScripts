import json
import os
import shutil
from typing import Generator, Optional, TypedDict
from urllib.parse import urlparse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from jet.scrapers.selenium import UrlScraper
from jet.scrapers.preprocessor import html_to_markdown, scrape_markdown, get_header_contents
from jet.scrapers.hrequests import request_url
from jet.transformers import to_snake_case
from jet.search import scrape_url
from jet.cache.redis import RedisConfigParams, RedisClient
from jet.vectors import SettingsManager, SettingsDict, QueryProcessor
from jet.actions import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.file import save_file
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.logger import logger


# Set working directory to script location
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

OUTPUT_DIR = os.path.join(file_dir, "generated", "scraped_urls")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

config = RedisConfigParams(
    port=3102
)

MODEL = "codellama"
SYSTEM_MESSAGE = "You are an AI assistant that follows instructions. You can understand and write code of any language, extract code from structured and unstructured content, and provide real-world usage examples. You can write clean, optimized, readable, and modular code. You follow best practices and correct syntax."
CHAT_OPTIONS: OllamaChatOptions = {
    "seed": 42,
    "num_ctx": 4096,
    "num_keep": 0,
    "num_predict": -1,
    "temperature": 0,
}
FINAL_MARKDOWN_TEMPLATE = "## System\n\n```\n{system}\n```\n\n## Prompt\n\n```\n{prompt}\n```\n\n## Response\n\n{response}"


class UrlItem(TypedDict):
    url: str
    container_selector: Optional[str]
    remove_selectors: Optional[list[str]]
    replace_selectors: Optional[list[str]]
    show_browser: Optional[bool]
    workflows: Optional[list]
    model: Optional[str]


def generate_filename(file_name: str, base_dir: str = None) -> str:
    output_dir = OUTPUT_DIR
    if base_dir:
        output_dir = os.path.join(output_dir, base_dir)

    return os.path.join(output_dir, file_name)


def scrape_urls(urls: list[str | UrlItem]):
    for url_item in urls:
        scrape_settings = {}
        if isinstance(url_item, dict):
            url = url_item.get('url')
            scrape_settings['container_selector'] = url_item.get(
                'container_selector', 'body')
            scrape_settings['remove_selectors'] = url_item.get(
                'remove_selectors', [])
            scrape_settings['replace_selectors'] = url_item.get(
                'replace_selectors', [])

            show_browser = url_item.get("show_browser", False)

        else:
            url = url_item

            show_browser = False

        html_str = scrape_url(url, config=config, show_browser=show_browser)

        markdown = scrape_markdown(html_str)
        # header_contents = markdown['headings']
        final_markdown = markdown["content"]

        # markdown = html_to_markdown(html_str, **scrape_settings)
        # headers_to_split_on = [
        #     ("#", "h1"),
        #     ("##", "h2"),
        #     ("###", "h3"),
        #     ("####", "h4"),
        #     ("#####", "h5"),
        #     ("######", "h6"),
        # ]
        # header_contents = get_header_contents(markdown, headers_to_split_on)
        # final_markdown = markdown

        result = {
            "url": url,
            "html": html_str,
            "markdown": final_markdown,
        }

        yield result

        if url_item.get('workflows'):
            result["queries"] = []

            for item in url_item.get('workflows'):
                parsed_url = urlparse(url)
                host_name = parsed_url.hostname

                query = item['template'].format(
                    context=final_markdown, query=item['query'])
                prompt = query + "\n\n" + final_markdown

                response = ""
                for chunk in call_ollama_chat(
                    prompt,
                    stream=True,
                    model=item['model'],
                    system=item.get('system', None),
                    options=CHAT_OPTIONS,
                    track={
                        "repo": "./aim-logs",
                        "experiment": "Code Scraper Test",
                        "run_name": host_name,
                        "format": FINAL_MARKDOWN_TEMPLATE,
                        "metadata": {
                            "type": "code_scraper",
                            "url_id": to_snake_case(url),
                            "url": url,
                        }
                    }
                ):
                    response += chunk

                query_result = {
                    "model": item['model'],
                    "system": item.get('system', None),
                    "prompt": prompt,
                    "response": response,
                }
                result["queries"].append(query_result)

                yield {
                    "url": url,
                    **query_result
                }


if __name__ == "__main__":
    # Anime Urls
    # urls = [
    #     "https://www.imdb.com/title/tt32812118/",
    #     "https://9animetv.to/watch/ill-become-a-villainess-who-goes-down-in-history-19334?ep=129043",
    #     "https://www.crunchyroll.com/series/GQWH0M17X/ill-become-a-villainess-who-goes-down-in-history",
    #     "https://zorotv.pro.in/ill-become-a-villainess-who-goes-down-in-history-episode-4/",
    # ]

    # LlamaIndex Urls
    urls = [
        {
            "url": "https://www.imdb.com/title/tt32812118",
            "show_browser": True,
            # "workflows": [
            #     {
            #         "model": "llama3.1",
            #         "query": (
            #             "Organize this scraped data."
            #         )
            #     }
            # ]
        }
    ]

    urls = [
        {
            # "url": "https://docs.llamaindex.ai/en/stable/examples/observability/AimCallback/",
            "url": "https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo/",
            "container_selector": '.md-content',
            "remove_selectors": [
                ".notice",
                '.clipboard-copy-txt',
            ],
            # "replace_selectors": [
            #     {".hl-python pre": "code"}
            # ],
            "workflows": [
                {
                    "model": "codellama",
                    "query": (
                        "Refactor this code as classes with types and typed dicts for readability, modularity, and reusability.\n"
                        "Add main function for real world usage examples.\n"
                        "Generated code should be complete and working with correct syntax.\n"
                        "Include logs and progress tracking if applicable\n"
                        "Add comments to explain each function and show installation instructions if dependencies are provided.\n"
                        "\nOutput only the Python code wrapped in a code block without additional information (use ```python)."
                    )
                }
            ]
        }
    ]

    urls = [
        {
            "url": "https://tomaarsen.github.io/SpanMarkerNER/notebooks/spacy_integration.html",
            "container_selector": '[role="main"]',
            "workflows": [
                {
                    "model": "llama3.1",
                    "query": (
                        "Write the python code based on the context.\n"
                        "Add main function for real world usage examples.\n"
                        "Respond with ONLY the Python code wrapped in a code block (```python) and DO NOT provide a reason."
                    ),
                    "template": (
                        "Context information is below. \n"
                        "---------------------\n"
                        "{context}"
                        "\n---------------------\n"
                        "Given the context information and not prior knowledge, "
                        "answer the query: {query}\n"
                    )
                }
            ]
        }
    ]

    stream_results = scrape_urls(urls)

    for stream_result in stream_results:
        url = stream_result["url"]
        html = stream_result.get("html")
        markdown = stream_result.get("markdown")
        model = stream_result.get("model")
        system = stream_result.get("system")
        prompt = stream_result.get("prompt")
        response = stream_result.get("response")

        base_dir = to_snake_case(url)

        if url and html and markdown:
            html_output_file = generate_filename("response.html", base_dir)
            md_output_file = generate_filename("response.md", base_dir)

            save_file(html, output_file=html_output_file)
            save_file(markdown, output_file=md_output_file)

        if url and model and prompt and response:
            json_output_file = generate_filename("response.json", base_dir)
            save_file({
                "url": url,
                "model": model,
                "system": system,
                "prompt": prompt,
                "response": response,
            }, output_file=json_output_file)

            # Save code if any
            extractor = MarkdownCodeExtractor()
            code_blocks = extractor.extract_code_blocks(response)
            for idx, code_block in enumerate(code_blocks):
                code_output_file = generate_filename(
                    f"code_{idx+1}{code_block['extension']}", base_dir)
                save_file(code_block['code'], output_file=code_output_file)
