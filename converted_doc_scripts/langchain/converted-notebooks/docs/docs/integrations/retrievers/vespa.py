from jet.logger import logger
from langchain_community.retrievers import VespaRetriever
from vespa.application import Vespa
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
# Vespa

>[Vespa](https://vespa.ai/) is a fully featured search engine and vector database. It supports vector search (ANN), lexical search, and search in structured data, all in the same query.

This notebook shows how to use `Vespa.ai` as a LangChain retriever.

In order to create a retriever, we use [pyvespa](https://pyvespa.readthedocs.io/en/latest/index.html) to
create a connection a `Vespa` service.
"""
logger.info("# Vespa")

# %pip install --upgrade --quiet  pyvespa


vespa_app = Vespa(url="https://doc-search.vespa.oath.cloud")

"""
This creates a connection to a `Vespa` service, here the Vespa documentation search service.
Using `pyvespa` package, you can also connect to a
[Vespa Cloud instance](https://pyvespa.readthedocs.io/en/latest/deploy-vespa-cloud.html)
or a local
[Docker instance](https://pyvespa.readthedocs.io/en/latest/deploy-docker.html).


After connecting to the service, you can set up the retriever:
"""
logger.info("This creates a connection to a `Vespa` service, here the Vespa documentation search service.")


vespa_query_body = {
    "yql": "select content from paragraph where userQuery()",
    "hits": 5,
    "ranking": "documentation",
    "locale": "en-us",
}
vespa_content_field = "content"
retriever = VespaRetriever(vespa_app, vespa_query_body, vespa_content_field)

"""
This sets up a LangChain retriever that fetches documents from the Vespa application.
Here, up to 5 results are retrieved from the `content` field in the `paragraph` document type,
using `doumentation` as the ranking method. The `userQuery()` is replaced with the actual query
passed from LangChain.

Please refer to the [pyvespa documentation](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa.html#Query)
for more information.

Now you can return the results and continue using the results in LangChain.
"""
logger.info("This sets up a LangChain retriever that fetches documents from the Vespa application.")

retriever.invoke("what is vespa?")

logger.info("\n\n[DONE]", bright=True)