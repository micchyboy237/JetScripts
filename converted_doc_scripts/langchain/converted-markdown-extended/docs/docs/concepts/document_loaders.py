from jet.logger import logger
from langchain_community.document_loaders.csv_loader import CSVLoader
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
# Document loaders
<span data-heading-keywords="document loader,document loaders"></span>

:::info[Prerequisites]

* [Document loaders API reference](/docs/how_to/#document-loaders)
:::

Document loaders are designed to load document objects. LangChain has hundreds of integrations with various data sources to load data from: Slack, Notion, Google Drive, etc.

## Integrations

You can find available integrations on the [Document loaders integrations page](/docs/integrations/document_loaders/).

## Interface

Documents loaders implement the [BaseLoader interface](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html).

Each DocumentLoader has its own specific parameters, but they can all be invoked in the same way with the `.load` method or `.lazy_load`.

Here's a simple example:
"""
logger.info("# Document loaders")


loader = CSVLoader(
    ...  # <-- Integration specific parameters here
)
data = loader.load()

"""
When working with large datasets, you can use the `.lazy_load` method:
"""
logger.info("When working with large datasets, you can use the `.lazy_load` method:")

for document in loader.lazy_load():
    logger.debug(document)

"""
## Related resources

Please see the following resources for more information:

* [How-to guides for document loaders](/docs/how_to/#document-loaders)
* [Document API reference](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)
* [Document loaders integrations](/docs/integrations/document_loaders/)
"""
logger.info("## Related resources")

logger.info("\n\n[DONE]", bright=True)