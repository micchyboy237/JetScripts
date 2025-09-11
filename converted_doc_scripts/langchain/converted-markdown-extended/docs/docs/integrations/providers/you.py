from jet.logger import logger
from langchain_community.retrievers.you import YouRetriever
from langchain_community.tools.you import YouSearchTool
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
# You

>[You](https://you.com/about) company provides an AI productivity platform.

## Retriever

See a [usage example](/docs/integrations/retrievers/you-retriever).
"""
logger.info("# You")


"""
## Tools

See a [usage example](/docs/integrations/tools/you).
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)