from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openrouter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OpenRouter

OpenRouter provides a standardized API to access many LLMs at the best price offered. You can find out more on their [homepage](https://openrouter.ai).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OpenRouter")

# %pip install llama-index-llms-openrouter

# !pip install llama-index


"""
## Call `chat` with ChatMessage List
You need to either set env var `OPENROUTER_API_KEY` or set api_key in the class constructor
"""
logger.info("## Call `chat` with ChatMessage List")

llm = OpenRouter(
    api_key="<your-api-key>",
    max_tokens=256,
    context_window=4096,
    model="gryphe/mythomax-l2-13b",
)

message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
logger.debug(resp)

"""
### Streaming
"""
logger.info("### Streaming")

message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])
for r in resp:
    logger.debug(r.delta, end="")

"""
## Call `complete` with Prompt
"""
logger.info("## Call `complete` with Prompt")

resp = llm.complete("Tell me a joke")
logger.debug(resp)

resp = llm.stream_complete("Tell me a story in 250 words")
for r in resp:
    logger.debug(r.delta, end="")

"""
## Model Configuration
"""
logger.info("## Model Configuration")

llm = OpenRouter(model="mistralai/mixtral-8x7b-instruct")

resp = llm.complete("Write a story about a dragon who can code in Rust")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)