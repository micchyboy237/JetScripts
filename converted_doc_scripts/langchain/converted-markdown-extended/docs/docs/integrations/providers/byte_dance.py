from jet.logger import logger
from langchain_community.document_loaders.larksuite import LarkSuiteDocLoader
from langchain_community.document_loaders.larksuite import LarkSuiteWikiLoader
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
# ByteDance

>[ByteDance](https://bytedance.com/) is a Chinese internet technology company.

## Installation and Setup

Get the access token.
You can find the access instructions [here](https://open.larksuite.com/document)


## Document Loaders

>[Lark Suite](https://www.larksuite.com/) is an enterprise collaboration platform
> developed by `ByteDance`.

### Lark Suite for Document

See a [usage example](/docs/integrations/document_loaders/larksuite/#load-from-document).
"""
logger.info("# ByteDance")


"""
### Lark Suite for Wiki

See a [usage example](/docs/integrations/document_loaders/larksuite/#load-from-wiki).
"""
logger.info("### Lark Suite for Wiki")


logger.info("\n\n[DONE]", bright=True)