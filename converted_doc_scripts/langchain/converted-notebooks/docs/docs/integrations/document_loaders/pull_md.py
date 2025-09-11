from jet.logger import logger
from langchain_pull_md.markdown_loader import PullMdLoader
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
sidebar_label: PullMdLoader
---

# PullMdLoader

Loader for converting URLs into Markdown using the pull.md service.

This package implements a [document loader](/docs/concepts/document_loaders/) for web content. Unlike traditional web scrapers, PullMdLoader can handle web pages built with dynamic JavaScript frameworks like React, Angular, or Vue.js, converting them into Markdown without local rendering.

## Overview
### Integration details

| Class | Package | Local | Serializable | JS Support |
| :--- | :--- | :---: | :---: | :---: |
| PullMdLoader | langchain-pull-md | ✅ | ✅ | ❌ |

## Setup

### Installation

```bash
pip install langchain-pull-md
```

### Initialization
"""
logger.info("# PullMdLoader")


loader = PullMdLoader(url="https://example.com")

"""
### Load
"""
logger.info("### Load")

documents = loader.load()

documents[0].metadata

"""
## Lazy Load

No lazy loading is implemented. `PullMdLoader` performs a real-time conversion of the provided URL into Markdown format whenever the `load` method is called.

## API reference:

- [GitHub](https://github.com/chigwell/langchain-pull-md)
- [PyPi](https://pypi.org/project/langchain-pull-md/)
"""
logger.info("## Lazy Load")

logger.info("\n\n[DONE]", bright=True)