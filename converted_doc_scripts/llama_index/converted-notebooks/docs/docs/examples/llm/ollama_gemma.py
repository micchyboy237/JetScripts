from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama_gemma.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ollama - Gemma

## Setup
First, follow the [readme](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance.

[Gemma](https://blog.google/technology/developers/gemma-open-models/): a family of lightweight, state-of-the-art open models built by Google DeepMind. Available in 2b and 7b parameter sizes

[Ollama](https://ollama.com/library/gemma): Support both 2b and 7b models

Note: `please install ollama>=0.1.26`
You can download pre-release version here [Ollama](https://github.com/ollama/ollama/releases/tag/v0.1.26)

When the Ollama app is running on your local machine:
- All of your local models are automatically served on localhost:11434
- Select your model when setting llm = Ollama(..., model="<model family>:<version>")
- Increase defaullt timeout (30 seconds) if needed setting Ollama(..., request_timeout=300.0)
- If you set llm = Ollama(..., model="<model family") without a version it will simply look for latest

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Ollama - Gemma")

# !pip install llama-index-llms-ollama

# !pip install llama-index


gemma_2b = Ollama(model="gemma:2b", request_timeout=30.0)
gemma_7b = Ollama(model="gemma:7b", request_timeout=30.0)

resp = gemma_2b.complete("Who is Paul Graham?")
logger.debug(resp)

resp = gemma_7b.complete("Who is Paul Graham?")
logger.debug(resp)

resp = gemma_2b.complete("Who is owning Tesla?")
logger.debug(resp)

resp = gemma_7b.complete("Who is owning Tesla?")
logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = gemma_7b.chat(messages)

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = gemma_7b.stream_complete("Who is Paul Graham?")

for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = gemma_7b.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)