from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import (
CallbackManager,
LlamaDebugHandler,
CBEventType,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/LlamaDebugHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Llama Debug Handler

Here we showcase the capabilities of our LlamaDebugHandler in logging events as we run queries
within LlamaIndex.

**NOTE**: This is a beta feature. The usage within different classes and the API interface
    for the CallbackManager and LlamaDebugHandler may change!

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Llama Debug Handler")

# %pip install llama-index-agent-openai
# %pip install llama-index-llms-ollama

# !pip install llama-index


"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Callback Manager Setup
"""
logger.info("## Callback Manager Setup")


llm = MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

"""
## Trigger the callback with a query
"""
logger.info("## Trigger the callback with a query")


index = VectorStoreIndex.from_documents(
    docs, callback_manager=callback_manager
)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

"""
## Explore the Debug Information

The callback manager will log several start and end events for the following types:
- CBEventType.LLM
- CBEventType.EMBEDDING
- CBEventType.CHUNKING
- CBEventType.NODE_PARSING
- CBEventType.RETRIEVE
- CBEventType.SYNTHESIZE 
- CBEventType.TREE
- CBEventType.QUERY

The LlamaDebugHandler provides a few basic methods for exploring information about these events
"""
logger.info("## Explore the Debug Information")

logger.debug(llama_debug.get_event_time_info(CBEventType.LLM))

event_pairs = llama_debug.get_llm_inputs_outputs()
logger.debug(event_pairs[0][0])
logger.debug(event_pairs[0][1].payload.keys())
logger.debug(event_pairs[0][1].payload["response"])

event_pairs = llama_debug.get_event_pairs(CBEventType.CHUNKING)
logger.debug(event_pairs[0][0].payload.keys())  # get first chunking start event
logger.debug(event_pairs[0][1].payload.keys())  # get first chunking end event

llama_debug.flush_event_logs()

logger.info("\n\n[DONE]", bright=True)