import json
from llama_index.core.llms import ChatMessage
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core.memory import VectorMemory
from jet.transformers.object import make_serializable
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/memory/vector_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vector Memory
#
# The vector memory module uses vector search (backed by a vector db) to retrieve relevant conversation items given a user input.
#
# This notebook shows you how to use the `VectorMemory` class. We show you how to use its individual functions. A typical usecase for vector memory is as a long-term memory storage of chat messages. You can

# ![VectorMemoryIllustration](https://d3ddy8balm3goa.cloudfront.net/llamaindex/vector-memory.excalidraw.svg)

# Initialize and Experiment with Memory Module
#
# Here we initialize a raw memory module and demonstrate its functions - to put and retrieve from ChatMessage objects.
#
# - Note that `retriever_kwargs` is the same args you'd specify on the `VectorIndexRetriever` or from `index.as_retriever(..)`.


vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=OllamaEmbedding(model_name="mxbai-embed-large"),
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
logger.newline()
logger.info("Response 1")
logger.success(json.dumps(make_serializable(msgs), indent=2))

vector_memory.reset()

# Now let's try resetting and trying again. This time, we'll add an assistant message. Note that user/assistant messages are bundled by default.

msgs = [
    ChatMessage.from_str("Jerry likes burgers.", "user"),
    ChatMessage.from_str("Bob likes apples.", "user"),
    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
    ChatMessage.from_str("Alice likes juice.", "user"),
]
vector_memory.set(msgs)

msgs = vector_memory.get("What does Bob like?")
logger.newline()
logger.info("Response 2")
logger.success(json.dumps(make_serializable(msgs), indent=2))

logger.info("\n\n[DONE]", bright=True)
