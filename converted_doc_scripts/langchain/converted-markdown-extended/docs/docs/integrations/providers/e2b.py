from jet.logger import logger
from langchain_community.tools import E2BDataAnalysisTool
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
# E2B

>[E2B](https://e2b.dev/) provides open-source secure sandboxes
> for AI-generated code execution. See more [here](https://github.com/e2b-dev).

## Installation and Setup

You have to install a python package:
"""
logger.info("# E2B")

pip install e2b_code_interpreter

"""
## Tool

See a [usage example](/docs/integrations/tools/e2b_data_analysis).
"""
logger.info("## Tool")


logger.info("\n\n[DONE]", bright=True)