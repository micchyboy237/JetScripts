from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import set_global_handler
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/LlamaDebugHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PromptLayer Handler
[PromptLayer](https://promptlayer.com) is an LLMOps tool to help manage prompts, check out the [features](https://docs.promptlayer.com/introduction). Currently we only support MLX for this integration.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and PromptLayer.
"""
logger.info("# PromptLayer Handler")

# !pip install llama-index
# !pip install promptlayer

"""
## Configure API keys
"""
logger.info("## Configure API keys")


# os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["PROMPTLAYER_API_KEY"] = "pl_..."

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Callback Manager Setup
"""
logger.info("## Callback Manager Setup")


set_global_handler("promptlayer", pl_tags=["paul graham", "essay"])

"""
## Trigger the callback with a query
"""
logger.info("## Trigger the callback with a query")


index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

"""
## Access [promptlayer.com](https://promptlayer.com) to see stats

![image.png](attachment:image.png)
"""
logger.info("## Access [promptlayer.com](https://promptlayer.com) to see stats")

logger.info("\n\n[DONE]", bright=True)