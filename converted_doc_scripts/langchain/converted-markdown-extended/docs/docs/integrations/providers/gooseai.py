from jet.logger import logger
from langchain_community.llms import GooseAI
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
# GooseAI

>[GooseAI](https://goose.ai) makes deploying NLP services easier and more accessible.
> `GooseAI` is a fully managed inference service delivered via API.
> With feature parity to other well known APIs, `GooseAI` delivers a plug-and-play solution
> for serving open source language models at the industry's best economics by simply
> changing 2 lines in your code.

## Installation and Setup

- Install the Python SDK with `pip install ollama`
- Get your GooseAI api key from this link [here](https://goose.ai/).
- Set the environment variable (`GOOSEAI_API_KEY`).
"""
logger.info("# GooseAI")

os.environ["GOOSEAI_API_KEY"] = "YOUR_API_KEY"

"""
## LLMs

See a [usage example](/docs/integrations/llms/gooseai).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)