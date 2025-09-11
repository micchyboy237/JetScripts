from jet.logger import logger
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
# Goodfire

[Goodfire](https://www.goodfire.ai/) is a research lab focused on AI safety and
interpretability.

## Installation and Setup
"""
logger.info("# Goodfire")

pip install langchain-goodfire

"""
## Chat models

See detail on available chat models [here](/docs/integrations/chat/goodfire).
"""
logger.info("## Chat models")

logger.info("\n\n[DONE]", bright=True)