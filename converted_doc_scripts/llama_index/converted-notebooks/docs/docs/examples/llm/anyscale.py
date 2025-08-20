from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anyscale import Anyscale
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/anyscale.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Anyscale

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Anyscale")

# %pip install llama-index-llms-anyscale

# !pip install llama-index


"""
## Call `chat` with ChatMessage List
You need to either set env var `ANYSCALE_API_KEY` or set api_key in the class constructor
"""
logger.info("## Call `chat` with ChatMessage List")

llm = Anyscale(api_key="<your-api-key>")

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

llm = Anyscale(model="codellama/CodeLlama-34b-Instruct-hf")

resp = llm.complete("Show me the c++ code to send requests to HTTP Server")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)