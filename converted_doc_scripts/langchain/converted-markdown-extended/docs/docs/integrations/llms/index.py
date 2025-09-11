from jet.logger import logger
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
keywords: [compatibility]
---

# LLMs

:::caution
You are currently on a page documenting the use of [text completion models](/docs/concepts/text_llms). Many of the latest and most popular models are [chat completion models](/docs/concepts/chat_models).

Unless you are specifically using more advanced prompting techniques, you are probably looking for [this page instead](/docs/integrations/chat/).
:::

[LLMs](/docs/concepts/text_llms) are language models that take a string as input and return a string as output.

:::info

If you'd like to write your own LLM, see [this how-to](/docs/how_to/custom_llm/).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/how_to/integrations/).

:::


<CategoryTable category="llms" />

## All LLMs

<IndexTable />
"""
logger.info("# LLMs")

logger.info("\n\n[DONE]", bright=True)