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
# SparkLLM

>[SparkLLM](https://xinghuo.xfyun.cn/spark) is a large-scale cognitive model independently developed by iFLYTEK.
It has cross-domain knowledge and language understanding ability by learning a large amount of texts, codes and images.
It can understand and perform tasks based on natural dialogue.

## Chat models

See a [usage example](/docs/integrations/chat/sparkllm).
"""
logger.info("# SparkLLM")


"""
## LLMs

See a [usage example](/docs/integrations/llms/sparkllm).
"""
logger.info("## LLMs")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/sparkllm)
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)