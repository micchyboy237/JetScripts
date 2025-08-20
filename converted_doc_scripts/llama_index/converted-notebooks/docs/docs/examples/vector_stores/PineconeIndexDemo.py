from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
MetadataFilter,
MetadataFilters,
FilterOperator,
FilterCondition,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import logging
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/PineconeIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pinecone Vector Store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Pinecone Vector Store")

# %pip install llama-index llama-index-vector-stores-pinecone


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
#### Creating a Pinecone Index
"""
logger.info("#### Creating a Pinecone Index")


os.environ["PINECONE_API_KEY"] = "..."
# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)



pc.create_index(
    name="quickstart",
    dimension=1536,
    metric="euclidean",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

pinecone_index = pc.Index("quickstart")

"""
#### Load documents, build the PineconeVectorStore and VectorStoreIndex
"""
logger.info("#### Load documents, build the PineconeVectorStore and VectorStoreIndex")


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


# if "OPENAI_API_KEY" not in os.environ:
#     raise EnvironmentError(f"Environment variable OPENAI_API_KEY is not set")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index

May take a minute or so for the index to be ready!
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
## Filtering

You can also fetch a list of nodes directly with filters.
"""
logger.info("## Filtering")


filter = MetadataFilters(
    filters=[
        MetadataFilter(
            key="file_path",
            value="/Users/loganmarkewich/giant_change/llama_index/docs/docs/examples/vector_stores/data/paul_graham/paul_graham_essay.txt",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)

"""
You can fetch nodes directly with the filters. The below will return all nodes that match the filter.
"""
logger.info("You can fetch nodes directly with the filters. The below will return all nodes that match the filter.")

nodes = vector_store.get_nodes(filters=filter, limit=100)
logger.debug(len(nodes))

"""
You can also fetch using top-k and filters.
"""
logger.info("You can also fetch using top-k and filters.")

query_engine = index.as_query_engine(similarity_top_k=2, filters=filter)
response = query_engine.query("What did the author do growing up?")
logger.debug(len(response.source_nodes))

logger.info("\n\n[DONE]", bright=True)