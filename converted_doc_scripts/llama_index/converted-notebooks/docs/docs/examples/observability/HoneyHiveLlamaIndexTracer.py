from honeyhive.utils.llamaindex_tracer import HoneyHiveLlamaIndexTracer
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
SimpleKeywordTableIndex,
StorageContext,
)
from llama_index.core import ComposableGraph
from llama_index.core import Settings
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/HoneyHiveLlamaIndexTracer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HoneyHive LlamaIndex Tracer

[HoneyHive](https://honeyhive.ai) is a platform that helps developers monitor, evaluate and continuously improve their LLM-powered applications.

The `HoneyHiveLlamaIndexTracer` is integrated with HoneyHive to help developers debug and analyze the execution flow of your LLM pipeline, or to let developers customize feedback on specific trace events to create evaluation or fine-tuning datasets from production.
"""
logger.info("# HoneyHive LlamaIndex Tracer")

# %pip install llama-index-llms-ollama

# from getpass import getpass

# if os.getenv("OPENAI_API_KEY") is None:
#     os.environ["OPENAI_API_KEY"] = getpass(
        "Paste your OllamaFunctionCallingAdapter key from:"
        " https://platform.openai.com/account/api-keys\n"
    )
# assert os.getenv("OPENAI_API_KEY", "").startswith(
    "sk-"
), "This doesn't look like a valid OllamaFunctionCallingAdapter API key"
logger.debug("OllamaFunctionCallingAdapter API key configured")

# from getpass import getpass

if os.getenv("HONEYHIVE_API_KEY") is None:
#     os.environ["HONEYHIVE_API_KEY"] = getpass(
        "Paste your HoneyHive key from:"
        " https://app.honeyhive.ai/settings/account\n"
    )
logger.debug("HoneyHive API key configured")

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index


"""
## Setup LLM
"""
logger.info("## Setup LLM")


Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0)

"""
## HoneyHive Callback Manager Setup

**Option 1**: Set Global Evaluation Handler
"""
logger.info("## HoneyHive Callback Manager Setup")


set_global_handler(
    "honeyhive",
    project="My LlamaIndex Project",
    name="My LlamaIndex Pipeline",
    api_key=os.environ["HONEYHIVE_API_KEY"],
)
hh_tracer = llama_index.core.global_handler

"""
**Option 2**: Manually Configure Callback Handler

Also configure a debugger handler for extra notebook visibility.
"""
logger.info("Also configure a debugger handler for extra notebook visibility.")

llama_debug = LlamaDebugHandler(print_trace_on_end=True)

hh_tracer = HoneyHiveLlamaIndexTracer(
    project="My LlamaIndex Project",
    name="My LlamaIndex Pipeline",
    api_key=os.environ["HONEYHIVE_API_KEY"],
)

callback_manager = CallbackManager([llama_debug, hh_tracer])

Settings.callback_manager = callback_manager

"""
## 1. Indexing

Download Data
"""
logger.info("## 1. Indexing")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(docs)

"""
## 2. Query Over Index
"""
logger.info("## 2. Query Over Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(response, sep="\n")

"""
## View HoneyHive Traces

When we are done tracing our events we can view them via [the HoneyHive platform](https://app.honeyhive.ai). Simply login to HoneyHive, go to your `My LlamaIndex Project` project, click the `Data Store` tab and view your `Sessions`.
"""
logger.info("## View HoneyHive Traces")

logger.info("\n\n[DONE]", bright=True)