from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# NVIDIA NIMs

The llama-index-postprocessor-nvidia-rerank` package contains LlamaIndex integrations building applications with models on 
NVIDIA NIM inference microservice. NIM supports models across domains like chat, embedding, and re-ranking models 
from the community as well as NVIDIA. These models are optimized by NVIDIA to deliver the best performance on NVIDIA 
accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single 
command on NVIDIA accelerated infrastructure.

NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing, 
NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud, 
giving enterprises ownership and full control of their IP and AI application.

NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog. 
At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.

# NVIDIA's Rerank connector

This example goes over how to use LlamaIndex to interact with the supported [NVIDIA Retrieval QA Ranking Model](https://build.nvidia.com/explore/retrieval) for [retrieval-augmented generation](https://developer.nvidia.com/blog/build-enterprise-retrieval-augmented-generation-apps-with-nvidia-retrieval-qa-embedding-model/) via the `NVIDIARerank` class.

# Reranking

Reranking is a critical piece of high accuracy, efficient retrieval pipelines.

Two important use cases:
- Combining results from multiple data sources
- Enhancing accuracy for single data sources

## Combining results from multiple sources

Consider a pipeline with data from a semantic store, such as VectorStoreIndex, as well as a BM25 store.

Each store is queried independently and returns results that the individual store considers to be highly relevant. Figuring out the overall relevance of the results is where reranking comes into play.

Follow along with the [Advanced - Hybrid Retriever + Re-Ranking](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#advanced-hybrid-retriever-re-ranking) use case, substitute [the reranker](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#re-ranker-setup) with -

## Installation
"""
logger.info("# NVIDIA NIMs")

# %pip install --upgrade --quiet llama-index-postprocessor-nvidia-rerank llama-index-llms-nvidia llama-index-readers-file

"""
## Setup

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Click on your model of choice.

3. Under Input select the Python tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.
"""
logger.info("## Setup")

# import getpass

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    logger.debug("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
#     nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

"""
## Working with API Catalog
"""
logger.info("## Working with API Catalog")


reranker = NVIDIARerank(top_n=4)

# !mkdir data
# !wget "https://www.dropbox.com/scl/fi/p33j9112y0ysgwg77fdjz/2021_Housing_Inventory.pdf?rlkey=yyok6bb18s5o31snjd2dxkxz3&dl=0" -O f"{GENERATED_DIR}/housing_data.pdf"

Settings.text_splitter = SentenceSplitter(chunk_size=500)

documents = SimpleDirectoryReader("./data").load_data()

Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

index = VectorStoreIndex.from_documents(documents)

Settings.llm = NVIDIA()

query_engine = index.as_query_engine(
    similarity_top_k=20, node_postprocessors=[reranker]
)

response = query_engine.query(
    "What was the net gain in housing units in the Mission in 2021?"
)
logger.debug(response)

"""
## Working with NVIDIA NIMs

In addition to connecting to hosted [NVIDIA NIMs](https://ai.nvidia.com), this connector can be used to connect to local microservice instances. This helps you take your applications local when necessary.

For instructions on how to setup local microservice instances, see https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/
"""
logger.info("## Working with NVIDIA NIMs")


reranker = NVIDIARerank(base_url="http://localhost:1976/v1")

logger.info("\n\n[DONE]", bright=True)