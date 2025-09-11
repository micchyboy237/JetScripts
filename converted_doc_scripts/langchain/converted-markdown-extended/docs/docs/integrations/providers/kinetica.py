from jet.logger import logger
from langchain_community.chat_models.kinetica import ChatKinetica
from langchain_community.document_loaders.kinetica_loader import KineticaLoader
from langchain_community.vectorstores import Kinetica
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
# Kinetica

[Kinetica](https://www.kinetica.com/) is a real-time database purpose built for enabling
analytics and generative AI on time-series & spatial data.

## Chat Model

The Kinetica LLM wrapper uses the [Kinetica SqlAssist
LLM](https://docs.kinetica.com/7.2/sql-gpt/concepts/) to transform natural language into
SQL to simplify the process of data retrieval.

See [Kinetica Language To SQL Chat Model](/docs/integrations/chat/kinetica) for usage.
"""
logger.info("# Kinetica")


"""
## Vector Store

The Kinetca vectorstore wrapper leverages Kinetica's native support for [vector
similarity search](https://docs.kinetica.com/7.2/vector_search/).

See [Kinetica Vectorstore API](/docs/integrations/vectorstores/kinetica) for usage.
"""
logger.info("## Vector Store")


"""
## Document Loader

The Kinetica Document loader can be used to load LangChain [Documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) from the
[Kinetica](https://www.kinetica.com/) database.

See [Kinetica Document Loader](/docs/integrations/document_loaders/kinetica) for usage
"""
logger.info("## Document Loader")


"""
## Retriever

The Kinetica Retriever can return documents given an unstructured query.

See [Kinetica VectorStore based Retriever](/docs/integrations/retrievers/kinetica) for usage
"""
logger.info("## Retriever")

logger.info("\n\n[DONE]", bright=True)