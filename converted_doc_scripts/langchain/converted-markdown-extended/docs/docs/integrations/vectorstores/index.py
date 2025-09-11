from jet.logger import logger
import EmbeddingTabs from "@theme/EmbeddingTabs";
import VectorStoreTabs from "@theme/VectorStoreTabs";
import os
import shutil
import { CategoryTable, IndexTable } from "@theme/FeatureTables";


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
sidebar_position: 0
sidebar_class_name: hidden
---

# Vector stores


A [vector store](/docs/concepts/vectorstores) stores [embedded](/docs/concepts/embedding_models) data and performs similarity search.

**Select embedding model:**


<EmbeddingTabs/>

**Select vector store:**


<VectorStoreTabs/>

<CategoryTable category="vectorstores" />

## All Vectorstores

<IndexTable />
"""
logger.info("# Vector stores")

logger.info("\n\n[DONE]", bright=True)