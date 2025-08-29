from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/GithubRepositoryReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Github Repo Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Github Repo Reader")

# %pip install llama-index-readers-github

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()

# %env OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# %env GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
github_token = os.environ.get("GITHUB_TOKEN")
owner = "jerryjliu"
repo = "llama_index"
branch = "main"

github_client = GithubClient(github_token=github_token, verbose=True)

documents = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
).load_data(branch=branch)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query(
    "What is the difference between VectorStoreIndex and SummaryIndex?",
    verbose=True,
)

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)