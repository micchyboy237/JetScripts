from jet.logger import logger
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
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
sidebar_label: YoutubeLoaderDL
---

# YoutubeLoaderDL

Loader for Youtube leveraging the `yt-dlp` library.

This package implements a [document loader](/docs/concepts/document_loaders/) for Youtube. In contrast to the [YoutubeLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.youtube.YoutubeLoader.html) of `langchain-community`, which relies on `pytube`, `YoutubeLoaderDL` is able to fetch YouTube metadata. `langchain-yt-dlp` leverages the robust `yt-dlp` library, providing a more reliable and feature-rich YouTube document loader.

## Overview
### Integration details

| Class | Package | Local | Serializable | JS Support |
| :--- | :--- | :---: | :---: | :---: |
| YoutubeLoader | langchain-yt-dlp | ✅ | ✅ | ❌ |

## Setup

### Installation

```bash
pip install langchain-yt-dlp
```

### Initialization
"""
logger.info("# YoutubeLoaderDL")


loader = YoutubeLoaderDL.from_youtube_url(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", add_video_info=True
)

"""
### Load
"""
logger.info("### Load")

documents = loader.load()

documents[0].metadata

"""
## Lazy Load

- No lazy loading is implemented

## API reference:

- [Github](https://github.com/aqib0770/langchain-yt-dlp)
- [PyPi](https://pypi.org/project/langchain-yt-dlp/)
"""
logger.info("## Lazy Load")

logger.info("\n\n[DONE]", bright=True)