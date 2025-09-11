from jet.logger import logger
from langchain_community.chat_models.llama_edge import LlamaEdgeChatService
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
# LlamaEdge

>[LlamaEdge](https://llamaedge.com/docs/intro/) is the easiest & fastest way to run customized
> and fine-tuned LLMs locally or on the edge.
>
>* Lightweight inference apps. `LlamaEdge` is in MBs instead of GBs
>* Native and GPU accelerated performance
>* Supports many GPU and hardware accelerators
>* Supports many optimized inference libraries
>* Wide selection of AI / LLM models



## Installation and Setup

See the [installation instructions](https://llamaedge.com/docs/user-guide/quick-start-command).

## Chat models

See a [usage example](/docs/integrations/chat/llama_edge).
"""
logger.info("# LlamaEdge")


logger.info("\n\n[DONE]", bright=True)