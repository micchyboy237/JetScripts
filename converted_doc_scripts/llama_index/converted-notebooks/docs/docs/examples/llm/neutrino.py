from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.neutrino import Neutrino
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/neutrino.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Neutrino AI

Neutrino lets you intelligently route queries to the best-suited LLM for the prompt, maximizing performance while optimizing for costs and latency.

Check us out at: <a href="https://www.neutrinoapp.com/">neutrinoapp.com</a>
Docs: <a href="https://docs.neutrinoapp.com/">docs.neutrinoapp.com</a>
Create an API key: <a href="https://platform.neutrinoapp.com/">platform.neutrinoapp.com</a>
"""
logger.info("# Neutrino AI")

# %pip install llama-index-llms-neutrino

# !pip install llama-index

"""
#### Set Neutrino API Key env variable
You can create an API key at: <a href="https://platform.neutrinoapp.com/">platform.neutrinoapp.com</a>
"""
logger.info("#### Set Neutrino API Key env variable")


os.environ["NEUTRINO_API_KEY"] = "<your-neutrino-api-key>"

"""
#### Using Your Router
A router is a collection of LLMs that you can route queries to. You can create a router in the Neutrino <a href="https://platform.neutrinoapp.com/">dashboard</a> or use the default router, which includes all supported models.
You can treat a router as a LLM.
"""
logger.info("#### Using Your Router")


llm = Neutrino(
)

response = llm.complete("In short, a Neutrino is")
logger.debug(f"Optimal model: {response.raw['model']}")
logger.debug(response)

message = ChatMessage(
    role="user",
    content="Explain the difference between statically typed and dynamically typed languages.",
)
resp = llm.chat([message])
logger.debug(f"Optimal model: {resp.raw['model']}")
logger.debug(resp)

"""
#### Streaming Responses
"""
logger.info("#### Streaming Responses")

message = ChatMessage(
    role="user", content="What is the approximate population of Mexico?"
)
resp = llm.stream_chat([message])
for i, r in enumerate(resp):
    if i == 0:
        logger.debug(f"Optimal model: {r.raw['model']}")
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)