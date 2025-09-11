from jet.logger import logger
from langchain_community.chat_models import ChatBaichuan
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.llms import BaichuanLLM
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
# Baichuan

>[Baichuan Inc.](https://www.baichuan-ai.com/) is a Chinese startup in the era of AGI,
> dedicated to addressing fundamental human needs: Efficiency, Health, and Happiness.


## Installation and Setup

Register and get an API key [here](https://platform.baichuan-ai.com/).

## LLMs

See a [usage example](/docs/integrations/llms/baichuan).
"""
logger.info("# Baichuan")


"""
## Chat models

See a [usage example](/docs/integrations/chat/baichuan).
"""
logger.info("## Chat models")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/baichuan).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)