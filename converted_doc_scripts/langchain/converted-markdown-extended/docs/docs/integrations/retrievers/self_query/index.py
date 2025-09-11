from jet.logger import logger
import DocCardList from "@theme/DocCardList";
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
---
sidebar-position: 0
---

# Self-querying retrievers

Learn about how the self-querying retriever works [here](/docs/how_to/self_query).


<DocCardList />
"""
logger.info("# Self-querying retrievers")

logger.info("\n\n[DONE]", bright=True)