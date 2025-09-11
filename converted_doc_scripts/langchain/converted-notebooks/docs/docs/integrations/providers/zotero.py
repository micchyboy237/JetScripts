from jet.logger import logger
from langchain_zotero_retriever.retrievers import ZoteroRetriever
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
# Zotero

[Zotero](https://www.zotero.org/) is an open source reference management system intended for managing bibliographic data and related research materials. You can connect to your personal library, as well as shared group libraries, via the [API](https://www.zotero.org/support/dev/web_api/v3/start). This retriever implementation utilizes [PyZotero](https://github.com/urschrei/pyzotero) to access libraries. 


## Installation

```bash
pip install pyzotero
```

## Retriever

See a [usage example](/docs/integrations/retrievers/zotero).

```python
```
"""
logger.info("# Zotero")

logger.info("\n\n[DONE]", bright=True)