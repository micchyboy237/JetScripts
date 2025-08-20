import asyncio
from jet.transformers.formatters import format_json
from airtrain import DatasetMetadata, upload_from_llama_nodes
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Node
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
Context,
Event,
StartEvent,
StopEvent,
Workflow,
step,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.readers.web import AsyncWebPageReader
import airtrain as at
import asyncio
import os
import shutil


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


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/airtrain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# AirtrainAI Cookbook

[Airtrain](https://www.airtrain.ai/) is a tool supporting unstructured/low-structured text datasets. It allows automated clustering, document classification, and more.

This cookbook showcases how to ingest and transform/enrich data with LlamaIndex and then upload the data to Airtrain for further processing and exploration.

## Installation & Setup
"""
logger.info("# AirtrainAI Cookbook")

# %pip install llama-index-embeddings-ollama==0.2.4
# %pip install llama-index-readers-web==0.2.2
# %pip install llama-index-readers-github==0.2.0

# %pip install airtrain-py[llama-index]

# import nest_asyncio

# nest_asyncio.apply()

"""
### API Key Setup

Set up the API keys that will be required to run the examples that follow.
The GitHub API token and MLX API key are only required for the example
'Usage with Readers/Embeddings/Splitters'. Instructions for getting a GitHub
access token are
[here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
while an MLX API key can be obtained
[here](https://platform.openai.com/api-keys).

To obtain your Airtrain API Key:
- Create an Airtrain Account by visting [here](https://app.airtrain.ai/api/auth/login)
- View "Settings" in the lower left, then go to "Billing" to sign up for a pro account or start a trial
- Copy your API key from the "Airtrain API Key" tab in "Billing"

Note that the Airtrain trial only allows ONE dataset at a time. As this notebook creates many, you may need
to delete the dataset in the Airtrain UI as you go along, to make space for another one.
"""
logger.info("### API Key Setup")


os.environ["GITHUB_TOKEN"] = "<your GitHub token>"
# os.environ["OPENAI_API_KEY"] = "<your OpenAi API key>"

os.environ["AIRTRAIN_API_KEY"] = "<your Airtrain API key>"

"""
## Example 1: Usage with Readers/Embeddings/Splitters

Some of the core abstractions in LlamaIndex are [Documents and Nodes](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/).
Airtrain's LlamaIndex integration allows you to create an Airtrain dataset using any iterable collection of either of these, via the
`upload_from_llama_nodes` function.

To illustrate the flexibility of this, we'll do both:
1. Create a dataset directly of documents. In this case whole pages from the [Sematic](https://docs.sematic.dev/) docs.
2. Use MLX embeddings and the `SemanticSplitterNodeParser` to split those documents into nodes, and create a dataset from those.
"""
logger.info("## Example 1: Usage with Readers/Embeddings/Splitters")



"""
The next step is to set up our reader. In this case we're using the GitHub reader, but that's just for illustrative purposes. Airtrain can ingest documents no matter what reader they came from originally.
"""
logger.info("The next step is to set up our reader. In this case we're using the GitHub reader, but that's just for illustrative purposes. Airtrain can ingest documents no matter what reader they came from originally.")

github_token = os.environ.get("GITHUB_TOKEN")
github_client = GithubClient(github_token=github_token, verbose=True)
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="sematic-ai",
    repo="sematic",
    use_parser=False,
    verbose=False,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".md",
        ],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
)
read_kwargs = dict(branch="main")

"""
Read the documents with the reader
"""
logger.info("Read the documents with the reader")

documents = reader.load_data(**read_kwargs)

"""
### Create dataset directly from documents

You can create an Airtrain dataset directly from these documents without doing any further
processing. In this case, Airtrain will automatically embed the documents for you before
generating further insights. Each row in the dataset will represent an entire markdown
document. Airtrain will automatically provide insights like semantic clustering of your
documents, allowing you to browse through the documents by looking at ones that cover similar
topics or uncovering subsets of documents that you might want to remove.

Though additional processing beyond basic document retrieval is not *required*, it is
*allowed*. You can enrich the documents with metadata, filter them, or manipulate them
in any way you like before uploading to Airtrain.
"""
logger.info("### Create dataset directly from documents")

result = at.upload_from_llama_nodes(
    documents,
    name="Sematic Docs Dataset: Whole Documents",
)
logger.debug(f"Uploaded {result.size} rows to '{result.name}'. View at: {result.url}")

"""
### Create dataset after splitting and embedding

If you wish to view a dataset oriented towards nodes within documents rather than whole documents, you can do that as well.
Airtrain will automatically create insights like a 2d PCA projection of your embedding vectors, so you can visually explore
the embedding space from which your RAG nodes will be retrieved. You can also click on individual rows and look at the ones
that are nearest to it in the full n-dimensional embedding space, to drill down further. Automated clusters and other insights
will also be generated to enrich and aid your exploration.

Here we'll use MLX embeddings and a `SemanticSplitterNodeParser` splitter, but you can use any other LlamaIndex tooling you
like to process your nodes before uploading to Airtrain. You can even skip embedding them yourself entirely, in which case
Airtrain will embed the nodes for you.
"""
logger.info("### Create dataset after splitting and embedding")

embed_model = MLXEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(documents)

"""
⚠️ **Note** ⚠️: If you are on an Airtrain trial and already created a whole-document dataset, you will need to delete it before uploading a new dataset.
"""

result = at.upload_from_llama_nodes(
    nodes,
    name="Sematic Docs, split + embedded",
)
logger.debug(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")

"""
## Example 2: Using the [Workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows) API

Since documents and nodes are the core abstractions the Airtrain integration works with, and these abstractions are
shared in LlamaIndex's workflows API, you can also use Airtrain as part of a broader workflow. Here we will illustrate
usage by scraping a few [Hacker News](https://news.ycombinator.com/) comment threads, but again you are not restricted
to web scraping workflows; any workflow producing documents or nodes will do.
"""
logger.info("## Example 2: Using the [Workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows) API")




"""
Specify the comment threads we'll be scraping from. The particular ones in this example were on or near the front page on September 30th, 2024. If
you wish to ingest from pages besides Hacker News, be aware that some sites have their content rendered client-side, in which case you might
want to use a reader like the `WholeSiteReader`, which uses a headless Chrome driver to render the page before returning the documents. For here
we'll use a page with server-side rendered HTML for simplicity.
"""
logger.info("Specify the comment threads we'll be scraping from. The particular ones in this example were on or near the front page on September 30th, 2024. If")

URLS = [
    "https://news.ycombinator.com/item?id=41694044",
    "https://news.ycombinator.com/item?id=41696046",
    "https://news.ycombinator.com/item?id=41693087",
    "https://news.ycombinator.com/item?id=41695756",
    "https://news.ycombinator.com/item?id=41666269",
    "https://news.ycombinator.com/item?id=41697137",
    "https://news.ycombinator.com/item?id=41695840",
    "https://news.ycombinator.com/item?id=41694712",
    "https://news.ycombinator.com/item?id=41690302",
    "https://news.ycombinator.com/item?id=41695076",
    "https://news.ycombinator.com/item?id=41669747",
    "https://news.ycombinator.com/item?id=41694504",
    "https://news.ycombinator.com/item?id=41697032",
    "https://news.ycombinator.com/item?id=41694025",
    "https://news.ycombinator.com/item?id=41652935",
    "https://news.ycombinator.com/item?id=41693979",
    "https://news.ycombinator.com/item?id=41696236",
    "https://news.ycombinator.com/item?id=41696434",
    "https://news.ycombinator.com/item?id=41688469",
    "https://news.ycombinator.com/item?id=41646782",
    "https://news.ycombinator.com/item?id=41689332",
    "https://news.ycombinator.com/item?id=41688018",
    "https://news.ycombinator.com/item?id=41668896",
    "https://news.ycombinator.com/item?id=41690087",
    "https://news.ycombinator.com/item?id=41679497",
    "https://news.ycombinator.com/item?id=41687739",
    "https://news.ycombinator.com/item?id=41686722",
    "https://news.ycombinator.com/item?id=41689138",
    "https://news.ycombinator.com/item?id=41691530",
]

"""
Next we'll define a basic event, as events are the standard way to pass data between steps in LlamaIndex workflows.
"""
logger.info("Next we'll define a basic event, as events are the standard way to pass data between steps in LlamaIndex workflows.")

class CompletedDocumentRetrievalEvent(Event):
    name: str
    documents: list[Node]

"""
After that we'll define the workflow itself. In our case, this will have one step to ingest the documents from the web, one to ingest them to Airtrain, and one to wrap up the workflow.
"""
logger.info("After that we'll define the workflow itself. In our case, this will have one step to ingest the documents from the web, one to ingest them to Airtrain, and one to wrap up the workflow.")

class IngestToAirtrainWorkflow(Workflow):
    @step
    async def ingest_documents(
        self, ctx: Context, ev: StartEvent
    ) -> CompletedDocumentRetrievalEvent | None:
        if not ev.get("urls"):
            return None
        reader = AsyncWebPageReader(html_to_text=True)
        async def run_async_code_4590c754():
            async def run_async_code_a17775e2():
                documents = await reader.aload_data(urls=ev.get("urls"))
                return documents
            documents = asyncio.run(run_async_code_a17775e2())
            logger.success(format_json(documents))
            return documents
        documents = asyncio.run(run_async_code_4590c754())
        logger.success(format_json(documents))
        return CompletedDocumentRetrievalEvent(
            name=ev.get("name"), documents=documents
        )

    @step
    async def ingest_documents_to_airtrain(
        self, ctx: Context, ev: CompletedDocumentRetrievalEvent
    ) -> StopEvent | None:
        dataset_meta = upload_from_llama_nodes(ev.documents, name=ev.name)
        return StopEvent(result=dataset_meta)

"""
Since the workflow API treats async code as a first-class citizen, we'll define an async `main` to drive the workflow.
"""
logger.info("Since the workflow API treats async code as a first-class citizen, we'll define an async `main` to drive the workflow.")

async def main() -> None:
    workflow = IngestToAirtrainWorkflow()
    async def async_func_2():
        result = await workflow.run(
            name="My HN Discussions Dataset",
            urls=URLS,
        )
        return result
    result = asyncio.run(async_func_2())
    logger.success(format_json(result))
    logger.debug(
        f"Uploaded {result.size} rows to {result.name}. View at: {result.url}"
    )

"""
Finally, we'll execute the async main using an asyncio event loop.

⚠️ **Note** ⚠️: If you are on an Airtrain trial and already ran examples above,
you will need to delete the resulting dataset(s) before uploading a new one.
"""
logger.info("Finally, we'll execute the async main using an asyncio event loop.")

asyncio.run(main())  # actually run the main & the workflow

logger.info("\n\n[DONE]", bright=True)