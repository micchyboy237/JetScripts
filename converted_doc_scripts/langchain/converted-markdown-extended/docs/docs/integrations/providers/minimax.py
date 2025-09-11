from jet.logger import logger
from langchain_community.chat_models import MiniMaxChat
from langchain_community.embeddings import MiniMaxEmbeddings
from langchain_community.llms import Minimax
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
# Minimax

>[Minimax](https://api.minimax.chat) is a Chinese startup that provides natural language processing models
> for companies and individuals.

## Installation and Setup
Get a [Minimax api key](https://api.minimax.chat/user-center/basic-information/interface-key) and set it as an environment variable (`MINIMAX_API_KEY`)
Get a [Minimax group id](https://api.minimax.chat/user-center/basic-information) and set it as an environment variable (`MINIMAX_GROUP_ID`)


## LLM

There exists a Minimax LLM wrapper, which you can access with
See a [usage example](/docs/integrations/llms/minimax).
"""
logger.info("# Minimax")


"""
## Chat Models

See a [usage example](/docs/integrations/chat/minimax)
"""
logger.info("## Chat Models")


"""
## Text Embedding Model

There exists a Minimax Embedding model, which you can access with
"""
logger.info("## Text Embedding Model")


logger.info("\n\n[DONE]", bright=True)