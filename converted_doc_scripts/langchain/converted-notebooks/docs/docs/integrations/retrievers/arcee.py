from jet.logger import logger
from langchain_community.retrievers import ArceeRetriever
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
# Arcee

>[Arcee](https://www.arcee.ai/about/about-us) helps with the development of the SLMs—small, specialized, secure, and scalable language models.

This notebook demonstrates how to use the `ArceeRetriever` class to retrieve relevant document(s) for Arcee's `Domain Adapted Language Models` (`DALMs`).

### Setup

Before using `ArceeRetriever`, make sure the Arcee API key is set as `ARCEE_API_KEY` environment variable. You can also pass the api key as a named parameter.
"""
logger.info("# Arcee")


retriever = ArceeRetriever(
    model="DALM-PubMed",
)

"""
### Additional Configuration

You can also configure `ArceeRetriever`'s parameters such as `arcee_api_url`, `arcee_app_url`, and `model_kwargs` as needed.
Setting the `model_kwargs` at the object initialization uses the filters and size as default for all the subsequent retrievals.
"""
logger.info("### Additional Configuration")

retriever = ArceeRetriever(
    model="DALM-PubMed",
    arcee_api_url="https://custom-api.arcee.ai",  # default is https://api.arcee.ai
    arcee_app_url="https://custom-app.arcee.ai",  # default is https://app.arcee.ai
    model_kwargs={
        "size": 5,
        "filters": [
            {
                "field_name": "document",
                "filter_type": "fuzzy_search",
                "value": "Einstein",
            }
        ],
    },
)

"""
### Retrieving documents
You can retrieve relevant documents from uploaded contexts by providing a query. Here's an example:
"""
logger.info("### Retrieving documents")

query = "Can AI-driven music therapy contribute to the rehabilitation of patients with disorders of consciousness?"
documents = retriever.invoke(query)

"""
### Additional parameters

Arcee allows you to apply `filters` and set the `size` (in terms of count) of retrieved document(s). Filters help narrow down the results. Here's how to use these parameters:
"""
logger.info("### Additional parameters")

filters = [
    {"field_name": "document", "filter_type": "fuzzy_search", "value": "Music"},
    {"field_name": "year", "filter_type": "strict_search", "value": "1905"},
]

documents = retriever.invoke(query, size=5, filters=filters)

logger.info("\n\n[DONE]", bright=True)