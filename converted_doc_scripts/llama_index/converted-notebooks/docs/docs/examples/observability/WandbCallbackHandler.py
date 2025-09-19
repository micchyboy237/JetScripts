from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.callbacks.wandb import WandbCallbackHandler
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
SimpleKeywordTableIndex,
StorageContext,
)
from llama_index.core import Settings
from llama_index.core import load_index_from_storage
from llama_index.core import set_global_handler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import LlamaDebugHandler
import llama_index.core
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/WandbCallbackHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Wandb Callback Handler

[Weights & Biases Prompts](https://docs.wandb.ai/guides/prompts) is a suite of LLMOps tools built for the development of LLM-powered applications.

The `WandbCallbackHandler` is integrated with W&B Prompts to visualize and inspect the execution flow of your index construction, or querying over your index and more. You can use this handler to persist your created indices as W&B Artifacts allowing you to version control your indices.
"""
logger.info("# Wandb Callback Handler")

# %pip install llama-index-callbacks-wandb
# %pip install llama-index-llms-ollama

# from getpass import getpass

# if os.getenv("OPENAI_API_KEY") is None:
#     os.environ["OPENAI_API_KEY"] = getpass(
        "Paste your OllamaFunctionCalling key from:"
        " https://platform.openai.com/account/api-keys\n"
    )
# assert os.getenv("OPENAI_API_KEY", "").startswith(
    "sk-"
), "This doesn't look like a valid OllamaFunctionCalling API key"
logger.debug("OllamaFunctionCalling API key configured")


"""
## Setup LLM
"""
logger.info("## Setup LLM")


Settings.llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0)

"""
## W&B Callback Manager Setup

**Option 1**: Set Global Evaluation Handler
"""
logger.info("## W&B Callback Manager Setup")


set_global_handler("wandb", run_args={"project": "llamaindex"})
wandb_callback = llama_index.core.global_handler

"""
**Option 2**: Manually Configure Callback Handler

Also configure a debugger handler for extra notebook visibility.
"""
logger.info("Also configure a debugger handler for extra notebook visibility.")

llama_debug = LlamaDebugHandler(print_trace_on_end=True)

run_args = dict(
    project="llamaindex",
)

wandb_callback = WandbCallbackHandler(run_args=run_args)

Settings.callback_manager = CallbackManager([llama_debug, wandb_callback])

"""
> After running the above cell, you will get the W&B run page URL. Here you will find a trace table with all the events tracked using [Weights and Biases' Prompts](https://docs.wandb.ai/guides/prompts) feature.

## 1. Indexing

Download Data
"""
logger.info("## 1. Indexing")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(docs)

"""
### 1.1 Persist Index as W&B Artifacts
"""
logger.info("### 1.1 Persist Index as W&B Artifacts")

wandb_callback.persist_index(index, index_name="simple_vector_store")

"""
### 1.2 Download Index from W&B Artifacts
"""
logger.info("### 1.2 Download Index from W&B Artifacts")


storage_context = wandb_callback.load_storage_context(
    artifact_url="ayut/llamaindex/simple_vector_store:v0"
)

index = load_index_from_storage(
    storage_context,
)

"""
## 2. Query Over Index
"""
logger.info("## 2. Query Over Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(response, sep="\n")

"""
## Close W&B Callback Handler

When we are done tracking our events we can close the wandb run.
"""
logger.info("## Close W&B Callback Handler")

wandb_callback.finish()

logger.info("\n\n[DONE]", bright=True)