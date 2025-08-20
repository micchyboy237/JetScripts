from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.pipeshift import Pipeshift
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
# [Pipeshift](https://pipeshift.com)

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# [Pipeshift](https://pipeshift.com)")

# %pip install llama-index-llms-pipeshift

# %pip install llama-index

"""
## Basic Usage

Head on to the [models](https://dashboard.pipeshift.com/models) section of pipeshift dashboard to see the list of available models.

#### Call `complete` with a prompt
"""
logger.info("## Basic Usage")



llm = Pipeshift(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
)
res = llm.complete("supercars are ")

logger.debug(res)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are sales person at supercar showroom"
    ),
    ChatMessage(role="user", content="why should I pick porsche 911 gt3 rs"),
]
res = Pipeshift(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens=50
).chat(messages)

logger.debug(res)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = Pipeshift(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
resp = llm.stream_complete("porsche GT3 RS is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


llm = Pipeshift(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
messages = [
    ChatMessage(
        role="system", content="You are sales person at supercar showroom"
    ),
    ChatMessage(role="user", content="how fast can porsche gt3 rs it go?"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)