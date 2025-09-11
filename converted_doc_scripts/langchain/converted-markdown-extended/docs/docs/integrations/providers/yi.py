from jet.logger import logger
from langchain_community.chat_models import ChatYi
from langchain_community.llms import YiLLM
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
# 01.AI

>[01.AI](https://www.lingyiwanwu.com/en), founded by Dr. Kai-Fu Lee, is a global company at the forefront of AI 2.0. They offer cutting-edge large language models, including the Yi series, which range from 6B to hundreds of billions of parameters. 01.AI also provides multimodal models, an open API platform, and open-source options like Yi-34B/9B/6B and Yi-VL.

## Installation and Setup

Register and get an API key from either the China site [here](https://platform.lingyiwanwu.com/apikeys) or the global site [here](https://platform.01.ai/apikeys).

## LLMs

See a [usage example](/docs/integrations/llms/yi).
"""
logger.info("# 01.AI")


"""
## Chat models

See a [usage example](/docs/integrations/chat/yi).
"""
logger.info("## Chat models")


logger.info("\n\n[DONE]", bright=True)