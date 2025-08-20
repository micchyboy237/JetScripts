import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/litellm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LiteLLM

### LiteLLM supports 100+ LLM APIs (Anthropic, Replicate, Huggingface, TogetherAI, Cohere, etc.). [Complete List](https://docs.litellm.ai/docs/providers)

#### Call `complete` with a prompt

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LiteLLM")

# %pip install llama-index-llms-litellm

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

message = ChatMessage(role="user", content="Hey! how's it going?")

llm = LiteLLM("gpt-3.5-turbo")
chat_response = llm.chat([message])

llm = LiteLLM("command-nightly")
chat_response = llm.chat([message])


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = LiteLLM("gpt-3.5-turbo").chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Async
"""
logger.info("## Async")


llm = LiteLLM("gpt-3.5-turbo")
async def run_async_code_c3ecd675():
    async def run_async_code_a989c387():
        resp = llm.complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_a989c387())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_c3ecd675())
logger.success(format_json(resp))

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)