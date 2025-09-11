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
# Netmind

[Netmind AI](https://www.netmind.ai/) Build AI Faster, Smarter, and More Affordably.
Train, Fine-tune, Run Inference, and Scale with our Global GPU Network—Your all-in-one AI Engine.

This example goes over how to use LangChain to interact with Netmind AI models.

## Installation and Setup

```bash
pip install langchain-netmind
```

Get an Netmind api key and set it as an environment variable (`NETMIND_API_KEY`).  
Head to https://www.netmind.ai/ to sign up to Netmind and generate an API key.

## Chat Models

For more on Netmind chat models, visit the guide [here](/docs/integrations/chat/netmind)

## Embedding Model

For more on Netmind embedding models, visit the [guide](/docs/integrations/text_embedding/netmind)
"""
logger.info("# Netmind")

logger.info("\n\n[DONE]", bright=True)