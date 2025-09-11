from jet.logger import logger
from langchain_community.llms import CerebriumAI
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
# CerebriumAI

>[Cerebrium](https://docs.cerebrium.ai/cerebrium/getting-started/introduction)  is a serverless GPU infrastructure provider.
> It provides API access to several LLM models.

See the examples in the [CerebriumAI documentation](https://docs.cerebrium.ai/examples/langchain).

## Installation and Setup

- Install a python package:
"""
logger.info("# CerebriumAI")

pip install cerebrium

"""
- [Get an CerebriumAI api key](https://docs.cerebrium.ai/cerebrium/getting-started/installation) and set
  it as an environment variable (`CEREBRIUMAI_API_KEY`)


## LLMs

See a [usage example](/docs/integrations/llms/cerebriumai).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)