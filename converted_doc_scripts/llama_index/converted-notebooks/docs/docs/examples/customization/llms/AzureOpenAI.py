from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.azure_openai import AzureHuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
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

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/llms/AzureOllamaFunctionCallingAdapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Azure OllamaFunctionCallingAdapter

Azure openAI resources unfortunately differ from standard openAI resources as you can't generate embeddings unless you use an embedding model. The regions where these models are available can be found here: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models

Furthermore the regions that support embedding models unfortunately don't support the latest versions (<*>-003) of openAI models, so we are forced to use one region for embeddings and another for the text generation.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Azure OllamaFunctionCallingAdapter")

# %pip install llama-index-embeddings-azure-openai
# %pip install llama-index-llms-azure-openai

# !pip install llama-index


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
Here, we setup the embedding model (for retrieval) and llm (for text generation).
Note that you need not only model names (e.g. "text-embedding-ada-002"), but also model deployment names (the one you chose when deploying the model in Azure.
You must pass the deployment name as a parameter when you initialize `AzureOllamaFunctionCallingAdapter` and `HuggingFaceEmbedding`.
"""
logger.info("Here, we setup the embedding model (for retrieval) and llm (for text generation).")

api_key = "<api-key>"
azure_endpoint = "https://<your-resource-name>.openai.azure.com/"
api_version = "2023-07-01-preview"

llm = AzureOllamaFunctionCallingAdapter(
    model="gpt-35-turbo-16k",
    deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureHuggingFaceEmbedding(
    model="text-embedding-ada-002",
    deployment_name="my-custom-embedding",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(
    input_files=[".././Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)

query = "What is most interesting about this essay?"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

logger.debug(answer.get_formatted_sources())
logger.debug("query was:", query)
logger.debug("answer was:", answer)

logger.info("\n\n[DONE]", bright=True)