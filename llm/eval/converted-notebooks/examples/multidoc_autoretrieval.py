from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.async_utils import run_jobs
from jet.llm.ollama.base import Ollama
from llama_index.core import Document, ServiceContext
from llama_index.core.indices import SummaryIndex
from tqdm.asyncio import tqdm_asyncio
import asyncio
from copy import deepcopy
from llama_index.readers.github import GitHubRepositoryIssuesReader, GitHubIssuesClient
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Multidoc Autoretrieval Pack
#
# This is the LlamaPack version of our structured hierarchical retrieval guide in the [core repo](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.html).

# Setup and Download Data
#
# In this section, we'll load in LlamaIndex Github issues.

# %pip install llama-index-readers-github
# %pip install llama-index-vector-stores-weaviate
# %pip install llama-index-llms-ollama


nest_asyncio.apply()


os.environ["GITHUB_TOKEN"] = ""


github_client = GitHubIssuesClient()
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="run-llama",
    repo="llama_index",
    verbose=True,
)

orig_docs = loader.load_data()

limit = 100

docs = []
for idx, doc in enumerate(orig_docs):
    doc.metadata["index_id"] = doc.id_
    if idx >= limit:
        break
    docs.append(doc)


async def aprocess_doc(doc, include_summary: bool = True):
    """Process doc."""
    print(f"Processing {doc.id_}")
    metadata = doc.metadata

    date_tokens = metadata["created_at"].split("T")[0].split("-")
    year = int(date_tokens[0])
    month = int(date_tokens[1])
    day = int(date_tokens[2])

    assignee = "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
    size = ""
    if len(doc.metadata["labels"]) > 0:
        size_arr = [l for l in doc.metadata["labels"] if "size:" in l]
        size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
    new_metadata = {
        "state": metadata["state"],
        "year": year,
        "month": month,
        "day": day,
        "assignee": assignee,
        "size": size,
        "index_id": doc.id_,
    }

    summary_index = SummaryIndex.from_documents([doc])
    query_str = "Give a one-sentence concise summary of this issue."
    query_engine = summary_index.as_query_engine(
        service_context=ServiceContext.from_defaults(llm=Ollama(
            model="llama3.2"))
    )
    summary_txt = str(query_engine.query(query_str))

    new_doc = Document(text=summary_txt, metadata=new_metadata)
    return new_doc


async def aprocess_docs(docs):
    """Process metadata on docs."""

    new_docs = []
    tasks = []
    for doc in docs:
        task = aprocess_doc(doc)
        tasks.append(task)

    new_docs = await run_jobs(tasks, show_progress=True, workers=5)

    return new_docs

new_docs = await aprocess_docs(docs)

new_docs[5].metadata

# Setup Weaviate Indices


auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(
    "https://<weaviate-cluster>.weaviate.network",
    auth_client_secret=auth_config,
)

doc_metadata_index_name = "LlamaIndex_auto"
doc_chunks_index_name = "LlamaIndex_AutoDoc"

client.schema.delete_class(doc_metadata_index_name)
client.schema.delete_class(doc_chunks_index_name)

# Setup Metadata Schema
#
# This is required for autoretrieval; we put this in the prompt.


vector_store_info = VectorStoreInfo(
    content_info="Github Issues",
    metadata_info=[
        MetadataInfo(
            name="state",
            description="Whether the issue is `open` or `closed`",
            type="string",
        ),
        MetadataInfo(
            name="year",
            description="The year issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="month",
            description="The month issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="day",
            description="The day issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="assignee",
            description="The assignee of the ticket",
            type="string",
        ),
        MetadataInfo(
            name="size",
            description="How big the issue is (XS, S, M, L, XL, XXL)",
            type="string",
        ),
    ],
)

# Download LlamaPack


MultiDocAutoRetrieverPack = download_llama_pack(
    "MultiDocAutoRetrieverPack", "./multidoc_autoretriever_pack"
)

pack = MultiDocAutoRetrieverPack(
    client,
    doc_metadata_index_name,
    doc_chunks_index_name,
    new_docs,
    docs,
    vector_store_info,
    auto_retriever_kwargs={
        "verbose": True,
        "similarity_top_k": 2,
        "empty_query_top_k": 10,
    },
    verbose=True,
)

# Run LlamaPack
#
# Now let's try the LlamaPack on some queries!

response = pack.run("Tell me about some issues on 12/11")
print(str(response))

response = pack.run("Tell me about some open issues related to agents")
print(str(response))

# Retriever-only
#
# We can also get the retriever module and just run that.

retriever = pack.get_modules()["recursive_retriever"]
nodes = retriever.retrieve("Tell me about some open issues related to agents")
print(f"Number of source nodes: {len(nodes)}")
nodes[0].node.metadata

logger.info("\n\n[DONE]", bright=True)
