from jet.logger import logger
from langchain_siliconflow import ChatSiliconFlow
from langchain_siliconflow import SiliconFlowEmbeddings
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
# langchain-siliconflow

This package contains the LangChain integration with SiliconFlow

## Installation
"""
logger.info("# langchain-siliconflow")

pip install -U langchain-siliconflow

"""
And you should configure credentials by setting the following environment variables:
"""
logger.info("And you should configure credentials by setting the following environment variables:")

export SILICONFLOW_API_KEY="your-api-key"

"""
You can set the following environment variable to use the `.cn` endpoint:
"""
logger.info("You can set the following environment variable to use the `.cn` endpoint:")

export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"

"""
## Chat Models

`ChatSiliconFlow` class exposes chat models from SiliconFlow.
"""
logger.info("## Chat Models")


llm = ChatSiliconFlow()
llm.invoke("Sing a ballad of LangChain.")

"""
## Embeddings

`SiliconFlowEmbeddings` class exposes embeddings from SiliconFlow.
"""
logger.info("## Embeddings")


embeddings = SiliconFlowEmbeddings()
embeddings.embed_query("What is the meaning of life?")

logger.info("\n\n[DONE]", bright=True)