from jet.logger import logger
from langchain_prolog import PrologConfig, PrologTool
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
# SWI-Prolog

SWI-Prolog offers a comprehensive free Prolog environment.

## Installation and Setup

Once SWI-Prolog has been installed, install lanchain-prolog using pip:
"""
logger.info("# SWI-Prolog")

pip install langchain-prolog

"""
## Tools

The `PrologTool` class allows the generation of langchain tools that use Prolog rules to generate answers.
"""
logger.info("## Tools")


"""
See a [usage example](/docs/integrations/tools/prolog_tool).

See the same guide for usage examples of `PrologRunnable`, which allows the generation
of LangChain runnables that use Prolog rules to generate answers.
"""
logger.info("See a [usage example](/docs/integrations/tools/prolog_tool).")

logger.info("\n\n[DONE]", bright=True)