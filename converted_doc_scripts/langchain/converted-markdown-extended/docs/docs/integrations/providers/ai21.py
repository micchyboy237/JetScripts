from jet.logger import logger
from langchain_ai21 import AI21ContextualAnswers
from langchain_ai21 import AI21LLM
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_ai21 import ChatAI21
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
# AI21 Labs

>[AI21 Labs](https://www.ai21.com/about) is a company specializing in Natural
> Language Processing (NLP), which develops AI systems
> that can understand and generate natural language.

This page covers how to use the `AI21` ecosystem within `LangChain`.

## Installation and Setup

- Get an AI21 api key and set it as an environment variable (`AI21_API_KEY`)
- Install the Python package:
"""
logger.info("# AI21 Labs")

pip install langchain-ai21

"""
## Chat models

### AI21 Chat

See a [usage example](/docs/integrations/chat/ai21).
"""
logger.info("## Chat models")


"""
## Deprecated features

:::caution The following features are deprecated.
:::

### AI21 LLM
"""
logger.info("## Deprecated features")


"""
### AI21 Contextual Answer
"""
logger.info("### AI21 Contextual Answer")


"""
## Text splitters

### AI21 Semantic Text Splitter
"""
logger.info("## Text splitters")


logger.info("\n\n[DONE]", bright=True)