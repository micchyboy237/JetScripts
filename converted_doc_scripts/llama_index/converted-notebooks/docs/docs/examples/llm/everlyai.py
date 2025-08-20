from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.everlyai import EverlyAI
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/everlyai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# EverlyAI

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# EverlyAI")

# %pip install llama-index-llms-everlyai

# !pip install llama-index


"""
## Call `chat` with ChatMessage List
You need to either set env var `EVERLYAI_API_KEY` or set api_key in the class constructor
"""
logger.info("## Call `chat` with ChatMessage List")

llm = EverlyAI(api_key="your-api-key")

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

logger.info("\n\n[DONE]", bright=True)