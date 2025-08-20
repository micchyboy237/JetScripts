from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureMLX
from llama_index.storage.chat_store.azure import AzureChatStore
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
# Demo: Azure Table Storage as a ChatStore

This guide shows you how to use our `AzureChatStore` abstraction which automatically persists chat histories to Azure Table Storage or CosmosDB.

<a href="https://colab.research.google.com/drive/1b_0JuVwWSXiLZZjeBAPr-u5_Y9b34Zcp?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Demo: Azure Table Storage as a ChatStore")

# %pip install llama-index
# %pip install llama-index-llms-azure-openai
# %pip install llama-index-storage-chat-store-azure

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


"""
# Define our models

In staying with the Azure theme, let's define our Azure MLX embedding and LLM models.
"""
logger.info("# Define our models")

Settings.llm = AzureMLXLlamaIndexLLMAdapter(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    deployment_name="gpt-4",
    api_key="",
    azure_endpoint="",
    api_version="2024-03-01-preview",
)

"""
We now define an `AzureChatStore`, memory and `SimpleChatEngine` to converse and store history in Azure Table Storage.
"""
logger.info("We now define an `AzureChatStore`, memory and `SimpleChatEngine` to converse and store history in Azure Table Storage.")


chat_store = AzureChatStore.from_account_and_key(
    account_name="",
    account_key="",
    chat_table_name="FranChat",
    metadata_table_name="FranChatMeta",
    metadata_partition_key="conversation1",
)

memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="conversation1",
)

chat_engine = SimpleChatEngine(
    memory=memory, llm=Settings.llm, prefix_messages=[]
)

"""
#### Test out a ChatEngine with memory backed by Azure Table Storage
"""
logger.info("#### Test out a ChatEngine with memory backed by Azure Table Storage")

response = chat_engine.chat("Hello, my name is Fran.")

display_response(response)

response = chat_engine.chat("What's my name again?")

display_response(response)

"""
#### Start a new conversation
"""
logger.info("#### Start a new conversation")

chat_store.metadata_partition_key = "conversation2"

memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="conversation2",
)

chat_engine = SimpleChatEngine(
    memory=memory, llm=Settings.llm, prefix_messages=[]
)

response = chat_engine.chat("What's in a name?")

display_response(response)

logger.info("\n\n[DONE]", bright=True)