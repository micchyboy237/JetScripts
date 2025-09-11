from jet.logger import logger
from langchain_community.document_loaders import DedocAPIFileLoader
from langchain_community.document_loaders import DedocFileLoader
from langchain_community.document_loaders import DedocPDFLoader
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
# Dedoc

>[Dedoc](https://dedoc.readthedocs.io) is an [open-source](https://github.com/ispras/dedoc)
library/service that extracts texts, tables, attached files and document structure
(e.g., titles, list items, etc.) from files of various formats.

`Dedoc` supports `DOCX`, `XLSX`, `PPTX`, `EML`, `HTML`, `PDF`, images and more.
Full list of supported formats can be found [here](https://dedoc.readthedocs.io/en/latest/#id1).

## Installation and Setup

### Dedoc library

You can install `Dedoc` using `pip`.
In this case, you will need to install dependencies,
please go [here](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html)
to get more information.
"""
logger.info("# Dedoc")

pip install dedoc

"""
### Dedoc API

If you are going to use `Dedoc` API, you don't need to install `dedoc` library.
In this case, you should run the `Dedoc` service, e.g. `Docker` container (please see
[the documentation](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker)
for more details):
"""
logger.info("### Dedoc API")

docker pull dedocproject/dedoc
docker run -p 1231:1231

"""
## Document Loader

* For handling files of any formats (supported by `Dedoc`), you can use `DedocFileLoader`:

    ```python
    ```

* For handling PDF files (with or without a textual layer), you can use `DedocPDFLoader`:

    ```python
    ```

* For handling files of any formats without library installation,
you can use `Dedoc API` with `DedocAPIFileLoader`:

    ```python
    ```

Please see a [usage example](/docs/integrations/document_loaders/dedoc) for more details.
"""
logger.info("## Document Loader")

logger.info("\n\n[DONE]", bright=True)