from jet.logger import logger
from langchain_community.chat_models import ChatSparkLLM
from langchain_community.embeddings import SparkLLMTextEmbeddings
from langchain_community.llms import SparkLLM
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
# iFlytek

>[iFlytek](https://www.iflytek.com) is a Chinese information technology company
> established in 1999. It creates voice recognition software and
> voice-based internet/mobile products covering education, communication,
> music, intelligent toys industries.


## Installation and Setup

- Get `SparkLLM` app_id, api_key and api_secret from [iFlyTek SparkLLM API Console](https://console.xfyun.cn/services/bm3) (for more info, see [iFlyTek SparkLLM Intro](https://xinghuo.xfyun.cn/sparkapi)).
- Install the Python package (not for the embedding models):
"""
logger.info("# iFlytek")

pip install websocket-client

"""
## LLMs

See a [usage example](/docs/integrations/llms/sparkllm).
"""
logger.info("## LLMs")


"""
## Chat models

See a [usage example](/docs/integrations/chat/sparkllm).
"""
logger.info("## Chat models")


"""
## Embedding models
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)