from __module_name__ import Chat__ModuleName__
from __module_name__ import __ModuleName__LLM
from __module_name__ import __ModuleName__VectorStore
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
# __ModuleName__

__ModuleName__ is a platform that offers...
"""
logger.info("# __ModuleName__")


logger.info("\n\n[DONE]", bright=True)