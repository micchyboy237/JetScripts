from jet.logger import logger
from llama_index.core.llms import ChatMessage
from llama_index.llms.cometapi import CometAPI
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/cometapi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CometAPI

CometAPI provides access to various state-of-the-art LLM models including GPT series, Claude series, Gemini series, and more through a unified Ollama-compatible interface. You can find out more on their [homepage](https://www.cometapi.com/).

Visit https://api.cometapi.com/console/token to sign up and get an API key.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# CometAPI")

# %pip install llama-index-llms-cometapi

# %pip install llama-index


"""
## Call `chat` with ChatMessage List
You need to either set env var `COMETAPI_API_KEY` or set api_key in the class constructor
"""
logger.info("## Call `chat` with ChatMessage List")


os.environ["COMETAPI_KEY"] = "<your-cometapi-key>"

api_key = os.getenv("COMETAPI_KEY")
llm = CometAPI(
    api_key=api_key,
    max_tokens=256,
    context_window=4096,
    model="gpt-5-chat-latest",
)


messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Say 'Hi' only!"),
]
resp = llm.chat(messages)
logger.debug(resp)

resp = llm.complete("Who is Kaiming He")

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

message = ChatMessage(role="user", content="Tell me what ResNet is")
resp = llm.stream_chat([message])
for r in resp:
    logger.debug(r.delta, end="")

resp = llm.stream_complete("Tell me about Large Language Models")

for r in resp:
    logger.debug(r.delta, end="")

"""
### Using Different Models

CometAPI supports various AI models including GPT, Claude, and Gemini series.
"""
logger.info("### Using Different Models")

claude_llm = CometAPI(
    api_key=api_key, model="claude-3-7-sonnet-latest", max_tokens=200
)

resp = claude_llm.complete("Explain deep learning briefly")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)