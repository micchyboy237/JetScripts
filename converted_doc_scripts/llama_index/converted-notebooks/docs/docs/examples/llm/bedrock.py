from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.bedrock import Bedrock
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/bedrock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Bedrock

**Deprecated**: Use [llama-index-llms-bedrock-converse](https://docs.llamaindex.ai/en/stable/examples/llm/bedrock_converse/) instead, the converse API is the recommended way to use Bedrock.

## Basic Usage

#### Call `complete` with a prompt

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Bedrock")

# %pip install llama-index-llms-bedrock

# !pip install llama-index


profile_name = "Your aws profile name"
resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).complete("Paul Graham is ")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
# Connect to Bedrock with Access Keys
"""
logger.info("# Connect to Bedrock with Access Keys")


llm = Bedrock(
    model="amazon.titan-text-express-v1",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    region_name="AWS Region to use, eg. us-east-1",
)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)