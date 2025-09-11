from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
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
sidebar_label: Upstage
---

# ChatUpstage

This notebook covers how to get started with Upstage chat models.

## Installation

Install `langchain-upstage` package.

```bash
pip install -U langchain-upstage
```

## Environment Setup

Make sure to set the following environment variables:

- `UPSTAGE_API_KEY`: Your Upstage API key from [Upstage console](https://console.upstage.ai/).

## Usage
"""
logger.info("# ChatUpstage")


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


chat = ChatUpstage()

chat.invoke("Hello, how are you?")

for m in chat.stream("Hello, how are you?"):
    logger.debug(m)

"""
## Chaining
"""
logger.info("## Chaining")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to French."),
        ("human", "Translate this sentence from English to French. {english_text}."),
    ]
)
chain = prompt | chat

chain.invoke({"english_text": "Hello, how are you?"})

logger.info("\n\n[DONE]", bright=True)