from jet.transformers.formatters import format_json
from bs4 import BeautifulSoup
from contextlib import contextmanager
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.agents import tool
from langchain.chains.qa_with_sources.loading import (
    BaseCombineDocumentsChain,
    load_qa_with_sources_chain,
)
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_experimental.autonomous_agents import AutoGPT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.async_api import async_playwright
from pydantic import Field
from typing import Optional
import asyncio
import faiss
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
## AutoGPT example finding Winning Marathon Times

* Implementation of https://github.com/Significant-Gravitas/Auto-GPT 
* With LangChain primitives (LLMs, PromptTemplates, VectorStores, Embeddings, Tools)
"""
logger.info("## AutoGPT example finding Winning Marathon Times")


# import nest_asyncio

# nest_asyncio.apply()

llm = ChatOllama(model="llama3.2")

"""
### Set up tools

* We'll set up an AutoGPT with a `search` tool, and `write-file` tool, and a `read-file` tool, a web browsing tool, and a tool to interact with a CSV file via a python REPL

Define any other `tools` you want to use below:
"""
logger.info("### Set up tools")


ROOT_DIR = "./data/"


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@tool
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(
            llm, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"


"""
**Browse a web page with PlayWright**
"""


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    logger.success(format_json(result))
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


"""
**Q&A Over a webpage**

Help the model ask more directed questions of web pages to avoid cluttering its memory
"""
logger.info(
    "Help the model ask more directed questions of web pages to avoid cluttering its memory")


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = (
        "Browse a webpage and retrieve the information relevant to the question."
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i: i + 4]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError


query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

"""
### Set up memory

* The memory here is used for the agents intermediate steps
"""
logger.info("### Set up memory")


embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

"""
### Setup model and AutoGPT

`Model set-up`
"""
logger.info("### Setup model and AutoGPT")

web_search = DuckDuckGoSearchRun()

tools = [
    web_search,
    WriteFileTool(root_dir="./data"),
    ReadFileTool(root_dir="./data"),
    process_csv,
    query_website_tool,
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
)

"""
### AutoGPT for Querying the Web
 
  
I've spent a lot of time over the years crawling data sources and cleaning data. Let's see if AutoGPT can help with this!

Here is the prompt for looking up recent boston marathon times and converting them to tabular form.
"""
logger.info("### AutoGPT for Querying the Web")

agent.run(
    [
        "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
    ]
)

logger.info("\n\n[DONE]", bright=True)
