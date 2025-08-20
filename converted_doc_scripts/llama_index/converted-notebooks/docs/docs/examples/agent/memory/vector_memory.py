from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import VectorMemory
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/memory/vector_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vector Memory

**NOTE:** This example of memory is deprecated in favor of the newer and more flexible `Memory` class. See the [latest docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/).

The vector memory module uses vector search (backed by a vector db) to retrieve relevant conversation items given a user input.

This notebook shows you how to use the `VectorMemory` class. We show you how to use its individual functions. A typical usecase for vector memory is as a long-term memory storage of chat messages. You can

![VectorMemoryIllustration](https://d3ddy8balm3goa.cloudfront.net/llamaindex/vector-memory.excalidraw.svg)

### Initialize and Experiment with Memory Module

Here we initialize a raw memory module and demonstrate its functions - to put and retrieve from ChatMessage objects.

- Note that `retriever_kwargs` is the same args you'd specify on the `VectorIndexRetriever` or from `index.as_retriever(..)`.
"""
logger.info("# Vector Memory")



vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=MLXEmbedding(),
    retriever_kwargs={"similarity_top_k": 1},
)


msgs = [
    ChatMessage.from_str("Jerry likes juice.", "user"),
    ChatMessage.from_str("Bob likes burgers.", "user"),
    ChatMessage.from_str("Alice likes apples.", "user"),
]

for m in msgs:
    vector_memory.put(m)

msgs = vector_memory.get("What does Jerry like?")
msgs

vector_memory.reset()

"""
Now let's try resetting and trying again. This time, we'll add an assistant message. Note that user/assistant messages are bundled by default.
"""
logger.info("Now let's try resetting and trying again. This time, we'll add an assistant message. Note that user/assistant messages are bundled by default.")

msgs = [
    ChatMessage.from_str("Jerry likes burgers.", "user"),
    ChatMessage.from_str("Bob likes apples.", "user"),
    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
    ChatMessage.from_str("Alice likes juice.", "user"),
]
vector_memory.set(msgs)

msgs = vector_memory.get("What does Bob like?")
msgs

logger.info("\n\n[DONE]", bright=True)