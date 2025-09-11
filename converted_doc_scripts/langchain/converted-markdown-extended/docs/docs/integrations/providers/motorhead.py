from jet.logger import logger
from langchain_community.memory import MotorheadMemory
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
# Motörhead

>[Motörhead](https://github.com/getmetal/motorhead) is a memory server implemented in Rust. It automatically handles incremental summarization in the background and allows for stateless applications.

## Installation and Setup

See instructions at [Motörhead](https://github.com/getmetal/motorhead) for running the server locally.


## Memory

See a [usage example](/docs/integrations/memory/motorhead_memory).
"""
logger.info("# Motörhead")


logger.info("\n\n[DONE]", bright=True)