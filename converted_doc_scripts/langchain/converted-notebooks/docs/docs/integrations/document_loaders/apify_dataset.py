from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama.embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.indexes import VectorstoreIndexCreator
from langchain_apify import ApifyDatasetLoader
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
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
# Apify Dataset

>[Apify Dataset](https://docs.apify.com/platform/storage/dataset) is a scalable append-only storage with sequential access built for storing structured web scraping results, such as a list of products or Google SERPs, and then export them to various formats like JSON, CSV, or Excel. Datasets are mainly used to save results of [Apify Actors](https://apify.com/store)—serverless cloud programs for various web scraping, crawling, and data extraction use cases.

This notebook shows how to load Apify datasets to LangChain.


## Prerequisites

You need to have an existing dataset on the Apify platform. This example shows how to load a dataset produced by the [Website Content Crawler](https://apify.com/apify/website-content-crawler).
"""
logger.info("# Apify Dataset")

# %pip install --upgrade --quiet langchain langchain-apify langchain-ollama

"""
First, import `ApifyDatasetLoader` into your source code:
"""
logger.info("First, import `ApifyDatasetLoader` into your source code:")


"""
Find your [Apify API token](https://console.apify.com/account/integrations) and [Ollama API key](https://platform.ollama.com/account/api-keys) and initialize these into environment variable:
"""
logger.info("Find your [Apify API token](https://console.apify.com/account/integrations) and [Ollama API key](https://platform.ollama.com/account/api-keys) and initialize these into environment variable:")


os.environ["APIFY_API_TOKEN"] = "your-apify-api-token"
# os.environ["OPENAI_API_KEY"] = "your-ollama-api-key"

"""
Then provide a function that maps Apify dataset record fields to LangChain `Document` format.

For example, if your dataset items are structured like this:

```json
{
    "url": "https://apify.com",
    "text": "Apify is the best web scraping and automation platform."
}
```

The mapping function in the code below will convert them to LangChain `Document` format, so that you can use them further with any LLM model (e.g. for question answering).
"""
logger.info(
    "Then provide a function that maps Apify dataset record fields to LangChain `Document` format.")

loader = ApifyDatasetLoader(
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"], metadata={
            "source": dataset_item["url"]}
    ),
)

data = loader.load()

"""
## An example with question answering

In this example, we use data from a dataset to answer a question.
"""
logger.info("## An example with question answering")


loader = ApifyDatasetLoader(
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

index = VectorstoreIndexCreator(
    vectorstore_cls=InMemoryVectorStore, embedding=OllamaEmbeddings(
        model="nomic-embed-text")
).from_loaders([loader])

llm = ChatOllama(model="llama3.2")

query = "What is Apify?"
result = index.query_with_sources(query, llm=llm)

logger.debug(result["answer"])
logger.debug(result["sources"])

logger.info("\n\n[DONE]", bright=True)
