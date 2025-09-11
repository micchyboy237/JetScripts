from jet.logger import logger
import EmbeddingTabs from "@theme/EmbeddingTabs";
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

# Embedding models


[Embedding models](/docs/concepts/embedding_models) create a vector representation of a piece of text.

This page documents integrations with various model providers that allow you to use embeddings in LangChain.


<EmbeddingTabs/>
"""
logger.info("# Embedding models")

embeddings.embed_query("Hello, world!")

"""
<CategoryTable category="text_embedding" />

## All embedding models

<IndexTable />
"""
logger.info("## All embedding models")

logger.info("\n\n[DONE]", bright=True)