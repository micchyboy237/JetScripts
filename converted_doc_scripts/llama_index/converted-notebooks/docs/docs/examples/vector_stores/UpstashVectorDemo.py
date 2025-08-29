from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import FilterOperator, FilterCondition
from llama_index.core.vector_stores import UpstashVectorStore
from llama_index.core.vector_stores.types import (
MetadataFilter,
MetadataFilters,
FilterOperator,
)
from llama_index.vector_stores.upstash import UpstashVectorStore
import openai
import os
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Upstash Vector Store

We're going to look at how to use LlamaIndex to interface with Upstash Vector!
"""
logger.info("# Upstash Vector Store")

# ! pip install -q llama-index upstash-vector


openai.api_key = "sk-..."

# ! mkdir -p 'data/paul_graham/'
# ! wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
Now, we can load the documents using the LlamaIndex SimpleDirectoryReader
"""
logger.info("Now, we can load the documents using the LlamaIndex SimpleDirectoryReader")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

logger.debug("# Documents:", len(documents))

"""
To create an index on Upstash, visit https://console.upstash.com/vector, create an index with 1536 dimensions and `Cosine` distance metric. Copy the URL and token below
"""
logger.info("To create an index on Upstash, visit https://console.upstash.com/vector, create an index with 1536 dimensions and `Cosine` distance metric. Copy the URL and token below")

vector_store = UpstashVectorStore(url="https://...", token="...")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Now we've successfully created an index and populated it with vectors from the essay! The data will take a second to index and then it'll be ready for querying.
"""
logger.info("Now we've successfully created an index and populated it with vectors from the essay! The data will take a second to index and then it'll be ready for querying.")

query_engine = index.as_query_engine()
res1 = query_engine.query("What did the author learn?")
logger.debug(textwrap.fill(str(res1), 100))

logger.debug("\n")

res2 = query_engine.query("What is the author's opinion on startups?")
logger.debug(textwrap.fill(str(res2), 100))

"""
### Metadata Filtering

You can pass `MetadataFilters` with your `VectorStoreQuery` to filter the nodes returned from Upstash vector store.
"""
logger.info("### Metadata Filtering")



vector_store = UpstashVectorStore(
    url=os.environ.get("UPSTASH_VECTOR_URL") or "",
    token=os.environ.get("UPSTASH_VECTOR_TOKEN") or "",
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="author", value="Marie Curie", operator=FilterOperator.EQ
        )
    ],
)

retriever = index.as_retriever(filters=filters)

retriever.retrieve("What is inception about?")

"""
We can also combine multiple `MetadataFilters` with `AND` or `OR` condition
"""
logger.info("We can also combine multiple `MetadataFilters` with `AND` or `OR` condition")


filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="theme",
            value=["Fiction", "Horror"],
            operator=FilterOperator.IN,
        ),
        MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.AND,
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("Harry Potter?")

logger.info("\n\n[DONE]", bright=True)