from jet.logger import logger
from langchain_upstage import UpstageDocumentParseLoader
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

# UpstageDocumentParseLoader

This notebook covers how to get started with `UpstageDocumentParseLoader`.

## Installation

Install `langchain-upstage` package.

```bash
pip install -U langchain-upstage
```

## Environment Setup

Make sure to set the following environment variables:

- `UPSTAGE_API_KEY`: Your Upstage API key. Read [Upstage developers document](https://developers.upstage.ai/docs/getting-started/quick-start) to get your API key.

> The previously used UPSTAGE_DOCUMENT_AI_API_KEY is deprecated. However, the key previously used in UPSTAGE_DOCUMENT_AI_API_KEY can now be used in UPSTAGE_API_KEY.

## Usage
"""
logger.info("# UpstageDocumentParseLoader")


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


file_path = "/PATH/TO/YOUR/FILE.pdf"
layzer = UpstageDocumentParseLoader(file_path, split="page")

docs = layzer.load()  # or layzer.lazy_load()

for doc in docs[:3]:
    logger.debug(doc)

logger.info("\n\n[DONE]", bright=True)