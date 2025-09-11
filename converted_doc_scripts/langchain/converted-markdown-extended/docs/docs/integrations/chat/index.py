from jet.logger import logger
import ChatModelTabs from "@theme/ChatModelTabs";
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

# Chat models

[Chat models](/docs/concepts/chat_models) are language models that use a sequence of [messages](/docs/concepts/messages) as inputs and return messages as outputs (as opposed to using plain text). These are generally newer models.

:::info

If you'd like to write your own chat model, see [this how-to](/docs/how_to/custom_chat_model/).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/how_to/integrations/).

:::


<ChatModelTabs overrideParams={{ollama: {model: "llama3.2"}}} />
"""
logger.info("# Chat models")

model.invoke("Hello, world!")

"""
## Featured Providers

:::info
While all these LangChain classes support the indicated advanced feature, you may have
to open the provider-specific documentation to learn which hosted models or backends support
the feature.
:::


<CategoryTable category="chat" />

## All chat models

<IndexTable />
"""
logger.info("## Featured Providers")

logger.info("\n\n[DONE]", bright=True)