from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import get_response_synthesizer
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from pathlib import Path
import logging
import os
import requests
import shutil
import sys
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/auto_vs_recursive_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Comparing Methods for Structured Retrieval (Auto-Retrieval vs. Recursive Retrieval)

In a naive RAG system, the set of input documents are then chunked, embedded, and dumped to a vector database collection. Retrieval would just fetch the top-k documents by embedding similarity.

This can fail if the set of documents is large - it can be hard to disambiguate raw chunks, and you're not guaranteed to filter for the set of documents that contain relevant context.

In this guide we explore **structured retrieval** - more advanced query algorithms that take advantage of structure within your documents for higher-precision retrieval. We compare the following two methods:

- **Metadata Filters + Auto-Retrieval**: Tag each document with the right set of metadata. During query-time, use auto-retrieval to infer metadata filters along with passing through the query string for semantic search.
- **Store Document Hierarchies (summaries -> raw chunks) + Recursive Retrieval**: Embed document summaries and map that to the set of raw chunks for each document. During query-time, do recursive retrieval to first fetch summaries before fetching documents.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Comparing Methods for Structured Retrieval (Auto-Retrieval vs. Recursive Retrieval)")

# %pip install llama-index-llms-ollama
# %pip install llama-index-vector-stores-weaviate

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

wiki_titles = ["Michael Jordan", "Elon Musk", "Richard Branson", "Rihanna"]
wiki_metadatas = {
    "Michael Jordan": {
        "category": "Sports",
        "country": "United States",
    },
    "Elon Musk": {
        "category": "Business",
        "country": "United States",
    },
    "Richard Branson": {
        "category": "Business",
        "country": "UK",
    },
    "Rihanna": {
        "category": "Music",
        "country": "Barbados",
    },
}



for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

docs_dict = {}
for wiki_title in wiki_titles:
    doc = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()[0]

    doc.metadata.update(wiki_metadatas[wiki_title])
    docs_dict[wiki_title] = doc



llm = OllamaFunctionCalling(model="llama3.2")
callback_manager = CallbackManager([LlamaDebugHandler()])
splitter = SentenceSplitter(chunk_size=256)

"""
## Metadata Filters + Auto-Retrieval

In this approach, we tag each Document with metadata (category, country), and store in a Weaviate vector db.

During retrieval-time, we then perform "auto-retrieval" to infer the relevant set of metadata filters.
"""
logger.info("## Metadata Filters + Auto-Retrieval")


auth_config = weaviate.AuthApiKey(api_key="<api_key>")
client = weaviate.Client(
    "https://llama-index-test-v0oggsoz.weaviate.network",
    auth_client_secret=auth_config,
)


client.schema.delete_class("LlamaIndex")


vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

class_schema = client.schema.get("LlamaIndex")
display(class_schema)

index = VectorStoreIndex(
    [],
    storage_context=storage_context,
    transformations=[splitter],
    callback_manager=callback_manager,
)

for wiki_title in wiki_titles:
    index.insert(docs_dict[wiki_title])



vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports, Entertainment,"
                " Business, Music]"
            ),
        ),
        MetadataInfo(
            name="country",
            type="str",
            description=(
                "Country of the celebrity, one of [United States, Barbados,"
                " Portugal]"
            ),
        ),
    ],
)
retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    llm=llm,
    callback_manager=callback_manager,
    max_top_k=10000,
)

nodes = retriever.retrieve(
    "Tell me about a celebrity from the United States, set top k to 10000"
)

logger.debug(f"Number of nodes: {len(nodes)}")
for node in nodes[:10]:
    logger.debug(node.node.get_content())

nodes = retriever.retrieve(
    "Tell me about the childhood of a popular sports celebrity in the United"
    " States"
)
for node in nodes:
    logger.debug(node.node.get_content())

nodes = retriever.retrieve(
    "Tell me about the college life of a billionaire who started at company at"
    " the age of 16"
)
for node in nodes:
    logger.debug(node.node.get_content())

nodes = retriever.retrieve("Tell me about the childhood of a UK billionaire")
for node in nodes:
    logger.debug(node.node.get_content())

"""
## Build Recursive Retriever over Document Summaries
"""
logger.info("## Build Recursive Retriever over Document Summaries")


nodes = []
vector_query_engines = {}
vector_retrievers = {}

for wiki_title in wiki_titles:
    vector_index = VectorStoreIndex.from_documents(
        [docs_dict[wiki_title]],
        transformations=[splitter],
        callback_manager=callback_manager,
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    vector_query_engines[wiki_title] = vector_query_engine
    vector_retrievers[wiki_title] = vector_index.as_retriever()

    out_path = Path("summaries") / f"{wiki_title}.txt"
    if not out_path.exists():
        summary_index = SummaryIndex.from_documents(
            [docs_dict[wiki_title]], callback_manager=callback_manager
        )

        summarizer = summary_index.as_query_engine(
            response_mode="tree_summarize", llm=llm
        )
        response = summarizer.query(
                f"Give me a summary of {wiki_title}"
            )
        logger.success(format_json(response))

        wiki_summary = response.response
        Path("summaries").mkdir(exist_ok=True)
        with open(out_path, "w") as fp:
            fp.write(wiki_summary)
    else:
        with open(out_path, "r") as fp:
            wiki_summary = fp.read()

    logger.debug(f"**Summary for {wiki_title}: {wiki_summary}")
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)

top_vector_index = VectorStoreIndex(
    nodes, transformations=[splitter], callback_manager=callback_manager
)
top_vector_retriever = top_vector_index.as_retriever(similarity_top_k=1)


recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": top_vector_retriever, **vector_retrievers},
    verbose=True,
)

nodes = recursive_retriever.retrieve(
    "Tell me about a celebrity from the United States"
)
for node in nodes:
    logger.debug(node.node.get_content())

nodes = recursive_retriever.retrieve(
    "Tell me about the childhood of a billionaire who started at company at"
    " the age of 16"
)
for node in nodes:
    logger.debug(node.node.get_content())

logger.info("\n\n[DONE]", bright=True)