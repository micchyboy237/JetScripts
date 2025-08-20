from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vercel_ai_gateway import VercelAIGateway
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/vercel-ai-gateway.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vercel AI Gateway

The AI Gateway is a proxy service from Vercel that routes model requests to various AI providers. It offers a unified API to multiple providers and gives you the ability to set budgets, monitor usage, load-balance requests, and manage fallbacks. You can find out more from their [docs](https://vercel.com/docs/ai-gateway)

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Vercel AI Gateway")

# %pip install llama-index-llms-vercel-ai-gateway

# !pip install llama-index


llm = VercelAIGateway(
    model="anthropic/claude-4-sonnet",
    max_tokens=64000,
    context_window=200000,
    api_key="your-api-key",
    default_headers={
        "http-referer": "https://myapp.com/",  # Optional: Your app URL
        "x-title": "My App",  # Optional: Your app name
    },
)

logger.debug(llm.model)

"""
## Call `chat` with ChatMessage List
You need to either set env var `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` or set api_key in the class constructor
"""
logger.info("## Call `chat` with ChatMessage List")

llm = VercelAIGateway(
    api_key="pBiuCWfswZCDxt8D50DSoBfU",
    max_tokens=64000,
    context_window=200000,
    model="anthropic/claude-4-sonnet",
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

llm = VercelAIGateway(
    model="anthropic/claude-4-sonnet",
    api_key="pBiuCWfswZCDxt8D50DSoBfU",
)

resp = llm.complete("Write a story about a dragon who can code in Rust")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)