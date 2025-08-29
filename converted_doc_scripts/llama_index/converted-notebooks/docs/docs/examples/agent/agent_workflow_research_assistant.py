import asyncio
from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_function_calling_llm_adapter import OllamaFunctionCallingAdapter
from jet.search.adapters.custom_browser_tool_spec import CustomBrowserToolSpec
from jet.search.adapters.searxng_llama_index_tool import SearXNGSearchToolSpec
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.tools.agentql import AgentQLBrowserToolSpec
# from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.playwright.base import PlaywrightToolSpec
from llama_index.core.tools import FunctionTool
from bs4 import BeautifulSoup
import os
import shutil
import logging
from typing import Dict, Optional

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)

logger.info(
    "Agent Workflow + Research Assistant with Playwright and BeautifulSoup")


def extract_metadata(page_content: str) -> Dict[str, Optional[str]]:
    """
    Extract metadata from HTML content using BeautifulSoup.

    Args:
        page_content (str): HTML content of the page.

    Returns:
        Dict[str, Optional[str]]: Extracted metadata (title, author, date, etc.).
    """
    try:
        soup = BeautifulSoup(page_content, 'html.parser')

        # Extract title
        title_tag = soup.find('title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else None

        # Extract author
        author_tag = soup.find(['meta'], attrs={'name': 'author'}) or \
            soup.find(['span', 'div'],
                      class_=lambda x: x and 'author' in x.lower())
        author = author_tag.get('content') or author_tag.get_text(
            strip=True) if author_tag else None

        # Extract publishing date
        date_tag = soup.find(['meta'], attrs={'name': 'date'}) or \
            soup.find(['time', 'span', 'div'],
                      class_=lambda x: x and 'date' in x.lower())
        date = date_tag.get('content') or date_tag.get_text(
            strip=True) if date_tag else None

        # Extract journal name
        journal_tag = soup.find(['meta'], attrs={'name': 'journal'}) or \
            soup.find(['span', 'div'],
                      class_=lambda x: x and 'journal' in x.lower())
        journal = journal_tag.get('content') or journal_tag.get_text(
            strip=True) if journal_tag else None

        # Extract volume and issue
        volume_tag = soup.find(
            ['span', 'div'], class_=lambda x: x and 'volume' in x.lower())
        issue_tag = soup.find(
            ['span', 'div'], class_=lambda x: x and 'issue' in x.lower())
        volume = volume_tag.get_text(strip=True) if volume_tag else None
        issue = issue_tag.get_text(strip=True) if issue_tag else None

        # Extract abstract
        abstract_tag = soup.find(['div', 'section', 'p'], class_=lambda x: x and 'abstract' in x.lower()) or \
            soup.find(['meta'], attrs={'name': 'description'})
        abstract = abstract_tag.get_text(strip=True) or abstract_tag.get(
            'content') if abstract_tag else None

        return {
            "title": title,
            "author": author,
            "publishing_date": date,
            "journal_name": journal,
            "volume_number": volume,
            "issue_number": issue,
            "abstract": abstract
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            "title": None,
            "author": None,
            "publishing_date": None,
            "journal_name": None,
            "volume_number": None,
            "issue_number": None,
            "abstract": None
        }


async def async_func_2():
    async_browser = await PlaywrightToolSpec.create_async_playwright_browser(
        headless=True
    )
    return async_browser

async_browser = asyncio.run(async_func_2())
logger.success(format_json(async_browser))

playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
playwright_tool_list = playwright_tool.to_tool_list()
playwright_agent_tool_list = [
    tool
    for tool in playwright_tool_list
    if tool.metadata.name in ["click", "get_current_page", "navigate_to"]
]

logger.info("Using Playwright, SearXNG, and metadata extraction tools.")

searxng_search_tool = [
    tool
    for tool in SearXNGSearchToolSpec().to_tool_list()
    if tool.metadata.name == "searxng_full_search"
]


custom_browser_tool = CustomBrowserToolSpec(async_browser=async_browser)

# Wrap extract_metadata as a LlamaIndex tool
metadata_tool = FunctionTool.from_defaults(
    fn=extract_metadata,
    name="extract_metadata",
    description="Extracts metadata (title, author, publishing date, journal name, volume number, issue number, abstract) from HTML content of a web page."
)

logger.info("Creating AgentWorkflow with imported tools.")

llm = OllamaFunctionCallingAdapter(model="llama3.2")

workflow = AgentWorkflow.from_tools_or_functions(
    playwright_agent_tool_list
    + searxng_search_tool
    + custom_browser_tool.to_tool_list(),
    # + [metadata_tool],
    llm=llm,
    system_prompt="You are an expert that can do browser automation, data extraction and text summarization for finding and extracting data from research resources.",
)


async def main():
    query = "What is the relationship between exercise and stress levels?"
    handler = workflow.run(
        user_msg="""
Use searxng_full_search to find URL resources on the web that are relevant to the research topic: {query}
Go through each resource found. For each different resource, use Playwright to navigate to the link to the resource, then use extract_web_data_from_browser to extract information, including the name of the resource, author name(s), link to the resource, publishing date, journal name, volume number, issue number, and the abstract.
Find more resources until there are two different resources that can be successfully extracted from.
""".format(query=query)
    )

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            logger.success(event.delta, flush=True)

    logger.info("\n\n[DONE]", bright=True)

if __name__ == "__main__":
    asyncio.run(main())
