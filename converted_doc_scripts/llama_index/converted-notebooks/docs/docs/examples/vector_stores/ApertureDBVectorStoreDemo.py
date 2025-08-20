from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import (
MetadataFilter,
FilterOperator,
FilterCondition,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.ApertureDB import ApertureDBVectorStore
import logging
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
# ApertureDB as a Vector Store with LlamaIndex.

**Note: This example assumes access to an ApertureDB instance, and available APERTUREDB_KEY. Sign up for a [free account](https://cloud.aperturedata.io/), or consider a [local installation](https://docs.aperturedata.io/Setup/server/Local).**

This notebook has examples for using ApertureDB as a vector store, and use it to semantic search, within the LlamaIndex framework.

### Install dependencies with pip
"""
logger.info("# ApertureDB as a Vector Store with LlamaIndex.")

# %pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-ApertureDB

"""
### Download the data
"""
logger.info("### Download the data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Import the dependencies
"""
logger.info("### Import the dependencies")




logging.basicConfig(level=logging.ERROR)

"""
### Create ApertureDBVectorStore
"""
logger.info("### Create ApertureDBVectorStore")

adb_vector_store = ApertureDBVectorStore(dimensions=1536)

"""
### Add the data to the Vector Store.
This needs to be done only once, and the generated embeddings and the metadata can be repeatedly queried.
"""
logger.info("### Add the data to the Vector Store.")

storage_context = StorageContext.from_defaults(
    vector_store=adb_vector_store, graph_store=SimpleGraphStore()
)

documents: Document = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
index: VectorStoreIndex = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the Vector Store with a pure Semantic search.
"""
logger.info("### Query the Vector Store with a pure Semantic search.")

index = VectorStoreIndex.from_vector_store(vector_store=adb_vector_store)

query_engine = index.as_query_engine()


def run_queries(query_engine):
    query_str = [
        "What did the author do growing up?",
        "What did the author do after his time at Viaweb?",
    ]
    for qs in query_str:
        response = query_engine.query(qs)
        logger.debug(f"{qs=}")
        logger.debug(f"{response.response=}")
        logger.debug("===" * 20)


run_queries(query_engine)

"""
## Search with filtering

There may be cases and there may be a need to specify a filter based on the metadata, as it would lead to more accurate context for the LLM. LlamaIndex lets you specify these as conditions to apply in addition to the Natural Language based query.

### Add documents with custom metadata

Here, we insert documents with some important metadata, that would be used while querying.
"""
logger.info("## Search with filtering")

adb_filterable_vector_store = ApertureDBVectorStore(
    dimensions=1536, descriptor_set="filterbale_embeddings"
)

storage_context = StorageContext.from_defaults(
    vector_store=adb_filterable_vector_store, graph_store=SimpleGraphStore()
)

documents = [
    Document(
        text="The band Van Halen was formed in 1973. The band was named after the Van Halen brothers, Eddie Van Halen and Alex Van Halen. They also had"
        " a third member, David Lee Roth, who was the lead singer and Michael Anthony on the bass The band was known for their energetic performances and innovative guitar work.",
        metadata={"members_start_year": 1974, "members_end_year": 1985},
    ),
    Document(
        text="Roth left the band in 1985 to pursue a solo career. He was replaced by Sammy Hagar, who had previously been the lead singer of the band Montrose. Hagar's"
        " first album with Van Halen, 5150, was released in 1986 and was a commercial success. The band continued to release successful albums throughout the late 1980s and early 1990s.",
        metadata={"members_start_year": 1985, "members_end_year": 1996},
    ),
    Document(
        text="Former extreme vocalist Gary Cherone replaced Hagar in 1996. The band released the album Van Halen III in 1998, which was not as successful as their previous albums. Cherone left the band in 1999.",
        metadata={"members_start_year": 1996, "members_end_year": 1999},
    ),
]

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query engine with filter.

Based on the previously applied metadata, the queries here run some extra filtering, which gets translated into corresponding DB queries in the Vector Store.
"""
logger.info("### Query engine with filter.")


adb_filterable_vector_store = ApertureDBVectorStore(
    dimensions=1536, descriptor_set="filterbale_embeddings"
)

index = VectorStoreIndex.from_vector_store(
    vector_store=adb_filterable_vector_store
)

year_ranges = [(1974, 1985), (1985, 1996), (1996, 1999)]
for start_year, end_year in year_ranges:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="members_start_year",
                value=end_year - 1,
                operator=FilterOperator.LT,
            )
        ],
        condition=FilterCondition.AND,
    )

    query_engine = index.as_query_engine(filters=filters, similarity_top_k=3)
    response = query_engine.query(
        "Who have been the members of Van Halen? Just list their names."
    )

    logger.debug(f"{response.response=}, {len(response.source_nodes)=}")
    for i, source_node in enumerate(response.source_nodes):
        logger.debug(f"{i=}, {source_node=}")

logger.info("\n\n[DONE]", bright=True)