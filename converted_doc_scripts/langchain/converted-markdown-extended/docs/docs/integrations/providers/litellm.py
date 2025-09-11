from jet.logger import logger
from langchain_litellm import ChatLiteLLM
from langchain_litellm import ChatLiteLLMRouter
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
# LiteLLM

>[LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Ollama, Azure, Huggingface, Replicate, etc.

## Installation and setup
"""
logger.info("# LiteLLM")

pip install langchain-litellm

"""
## Chat Models
"""
logger.info("## Chat Models")



"""
See more detail in the guide [here](/docs/integrations/chat/litellm).

## API reference
For detailed documentation of all `ChatLiteLLM` and `ChatLiteLLMRouter` features and configurations head to the API reference: https://github.com/Akshay-Dongare/langchain-litellm
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)