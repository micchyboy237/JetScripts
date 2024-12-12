import os
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
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.logger import logger


# Set working directory to script location
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

output_dir = os.path.join(file_dir, "generated", "scraped_urls")
os.makedirs(output_dir, exist_ok=True)

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


def generate_filename(url: str, extension: str) -> str:
    # Convert URL to snake case
    snake_case_url = to_snake_case(url)
    return os.path.join(output_dir, f"{snake_case_url}{extension}")


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
        }

        if url_item.get('workflows'):
            result["queries"] = []

            for item in url_item.get('workflows'):
                parsed_url = urlparse(url)
                host_name = parsed_url.hostname

                model = MODEL
                query = item['query']
                prompt = query + "\n\n" + final_markdown
                logger.log("PROMPT:")
                logger.info(prompt)
                logger.debug("Generating response...")
                response = ""
                for chunk in call_ollama_chat(
                    prompt,
                    stream=True,
                    model=model,
                    system=SYSTEM_MESSAGE,
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
                    logger.success(chunk, flush=True)
                query_result = {
                    "system": SYSTEM_MESSAGE,
                    "prompt": prompt,
                    "response": response,
                }
                result["queries"].append(query_result)

                yield {
                    "url": url,
                    **result,
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

    stream_results = scrape_urls(urls)

    for stream_result in stream_results:
        url = stream_result["url"]
        html_str = stream_result["html"]
        system = stream_result["system"]
        prompt = stream_result["prompt"]
        response = stream_result["response"]

        final_markdown = FINAL_MARKDOWN_TEMPLATE.format(
            system=system,
            prompt=prompt,
            response=response,
        )

        html_output_file = generate_filename(url, ".html")
        md_output_file = generate_filename(url, ".md")

        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(stream_result["html"])
        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        logger.log("HTML", f"({len(stream_result['html'])})", "saved to:", html_output_file,
                   colors=["GRAY", "SUCCESS", "GRAY", "BRIGHT_SUCCESS"])
        logger.log("MD", f"({len(final_markdown)})", "saved to:", md_output_file,
                   colors=["GRAY", "SUCCESS", "GRAY", "BRIGHT_SUCCESS"])
