from jet.logger import logger
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader, PebbloSafeLoader
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
# Pebblo Safe DocumentLoader

> [Pebblo](https://daxa-ai.github.io/pebblo/) enables developers to safely load data and promote their Gen AI app to deployment without worrying about the organization’s compliance and security requirements. The project identifies semantic topics and entities found in the loaded data and summarizes them on the UI or a PDF report.

Pebblo has two components.

1. Pebblo Safe DocumentLoader for Langchain
1. Pebblo Server

This document describes how to augment your existing Langchain DocumentLoader with Pebblo Safe DocumentLoader to get deep data visibility on the types of Topics and Entities ingested into the Gen-AI Langchain application. For details on `Pebblo Server` see this [pebblo server](https://daxa-ai.github.io/pebblo/daemon) document.

Pebblo Safeloader enables safe data ingestion for Langchain `DocumentLoader`. This is done by wrapping the document loader call with `Pebblo Safe DocumentLoader`.

Note: To configure pebblo server on some url other that pebblo's default (localhost:8000) url, put the correct URL in `PEBBLO_CLASSIFIER_URL` env variable. This is configurable using the `classifier_url` keyword argument as well. Ref: [server-configurations](https://daxa-ai.github.io/pebblo/config)

#### How to Pebblo enable Document Loading?

Assume a Langchain RAG application snippet using `CSVLoader` to read a CSV document for inference.

Here is the snippet of Document loading using `CSVLoader`.
"""
logger.info("# Pebblo Safe DocumentLoader")


loader = CSVLoader("data/corp_sens_data.csv")
documents = loader.load()
logger.debug(documents)

"""
The Pebblo SafeLoader can be enabled with few lines of code change to the above snippet.
"""
logger.info("The Pebblo SafeLoader can be enabled with few lines of code change to the above snippet.")


loader = PebbloSafeLoader(
    CSVLoader("data/corp_sens_data.csv"),
    name="acme-corp-rag-1",  # App name (Mandatory)
    owner="Joe Smith",  # Owner (Optional)
    description="Support productivity RAG application",  # Description (Optional)
)
documents = loader.load()
logger.debug(documents)

"""
### Send semantic topics and identities to Pebblo cloud server

To send semantic data to pebblo-cloud, pass api-key to PebbloSafeLoader as an argument or alternatively, put the api-key in `PEBBLO_API_KEY` environment variable.
"""
logger.info("### Send semantic topics and identities to Pebblo cloud server")


loader = PebbloSafeLoader(
    CSVLoader("data/corp_sens_data.csv"),
    name="acme-corp-rag-1",  # App name (Mandatory)
    owner="Joe Smith",  # Owner (Optional)
    description="Support productivity RAG application",  # Description (Optional)
    # API key (Optional, can be set in the environment variable PEBBLO_API_KEY)
)
documents = loader.load()
logger.debug(documents)

"""
### Add semantic topics and identities to loaded metadata

To add semantic topics and sematic entities to metadata of loaded documents, set load_semantic to True as an argument or alternatively, define a new environment variable `PEBBLO_LOAD_SEMANTIC`, and setting it to True.
"""
logger.info("### Add semantic topics and identities to loaded metadata")


loader = PebbloSafeLoader(
    CSVLoader("data/corp_sens_data.csv"),
    name="acme-corp-rag-1",  # App name (Mandatory)
    owner="Joe Smith",  # Owner (Optional)
    description="Support productivity RAG application",  # Description (Optional)
    # API key (Optional, can be set in the environment variable PEBBLO_API_KEY)
    load_semantic=True,  # Load semantic data (Optional, default is False, can be set in the environment variable PEBBLO_LOAD_SEMANTIC)
)
documents = loader.load()
logger.debug(documents[0].metadata)

"""
### Anonymize the snippets to redact all PII details

Set `anonymize_snippets` to `True` to anonymize all personally identifiable information (PII) from the snippets going into VectorDB and the generated reports.

> Note: The _Pebblo Entity Classifier_ effectively identifies personally identifiable information (PII) and is continuously evolving. While its recall is not yet 100%, it is steadily improving.
> For more details, please refer to the [_Pebblo Entity Classifier docs_](https://daxa-ai.github.io/pebblo/entityclassifier/)
"""
logger.info("### Anonymize the snippets to redact all PII details")


loader = PebbloSafeLoader(
    CSVLoader("data/corp_sens_data.csv"),
    name="acme-corp-rag-1",  # App name (Mandatory)
    owner="Joe Smith",  # Owner (Optional)
    description="Support productivity RAG application",  # Description (Optional)
    anonymize_snippets=True,  # Whether to anonymize entities in the PDF Report (Optional, default=False)
)
documents = loader.load()
logger.debug(documents[0].metadata)

logger.info("\n\n[DONE]", bright=True)