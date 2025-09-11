from jet.logger import logger
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import SpreedlyLoader
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
# Spreedly

>[Spreedly](https://docs.spreedly.com/) is a service that allows you to securely store credit cards and use them to transact against any number of payment gateways and third party APIs. It does this by simultaneously providing a card tokenization/vault service as well as a gateway and receiver integration service. Payment methods tokenized by Spreedly are stored at `Spreedly`, allowing you to independently store a card and then pass that card to different end points based on your business requirements.

This notebook covers how to load data from the [Spreedly REST API](https://docs.spreedly.com/reference/api/v1/) into a format that can be ingested into LangChain, along with example usage for vectorization.

Note: this notebook assumes the following packages are installed: `ollama`, `chromadb`, and `tiktoken`.
"""
logger.info("# Spreedly")



"""
Spreedly API requires an access token, which can be found inside the Spreedly Admin Console.

This document loader does not currently support pagination, nor access to more complex objects which require additional parameters. It also requires a `resource` option which defines what objects you want to load.

Following resources are available:
- `gateways_options`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-supported-gateways)
- `gateways`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-created-gateways)
- `receivers_options`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-supported-receivers)
- `receivers`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-created-receivers)
- `payment_methods`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list)
- `certificates`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-certificates)
- `transactions`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list49)
- `environments`: [Documentation](https://docs.spreedly.com/reference/api/v1/#list-environments)
"""
logger.info("Spreedly API requires an access token, which can be found inside the Spreedly Admin Console.")

spreedly_loader = SpreedlyLoader(
    os.environ["SPREEDLY_ACCESS_TOKEN"], "gateways_options"
)

index = VectorstoreIndexCreator().from_loaders([spreedly_loader])
spreedly_doc_retriever = index.vectorstore.as_retriever()

spreedly_doc_retriever.invoke("CRC")

logger.info("\n\n[DONE]", bright=True)