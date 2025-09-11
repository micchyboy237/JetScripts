from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MyScale
from langchain_community.vectorstores import MyScale, MyScaleSettings
from langchain_text_splitters import CharacterTextSplitter
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
# MyScale

>[MyScale](https://docs.myscale.com/en/overview/) is a cloud-based database optimized for AI applications and solutions, built on the open-source [ClickHouse](https://github.com/ClickHouse/ClickHouse). 

This notebook shows how to use functionality related to the `MyScale` vector database.

## Setting up environments
"""
logger.info("# MyScale")

# %pip install --upgrade --quiet  clickhouse-connect langchain-community

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info("We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
if "OPENAI_API_BASE" not in os.environ:
#     os.environ["OPENAI_API_BASE"] = getpass.getpass("Ollama Base:")
if "MYSCALE_HOST" not in os.environ:
#     os.environ["MYSCALE_HOST"] = getpass.getpass("MyScale Host:")
if "MYSCALE_PORT" not in os.environ:
#     os.environ["MYSCALE_PORT"] = getpass.getpass("MyScale Port:")
if "MYSCALE_USERNAME" not in os.environ:
#     os.environ["MYSCALE_USERNAME"] = getpass.getpass("MyScale Username:")
if "MYSCALE_PASSWORD" not in os.environ:
#     os.environ["MYSCALE_PASSWORD"] = getpass.getpass("MyScale Password:")

"""
There are two ways to set up parameters for myscale index.

1. Environment Variables

    Before you run the app, please set the environment variable with `export`:
    `export MYSCALE_HOST='<your-endpoints-url>' MYSCALE_PORT=<your-endpoints-port> MYSCALE_USERNAME=<your-username> MYSCALE_PASSWORD=<your-password> ...`

    You can easily find your account, password and other info on our SaaS. For details please refer to [this document](https://docs.myscale.com/en/cluster-management/)

    Every attributes under `MyScaleSettings` can be set with prefix `MYSCALE_` and is case insensitive.

2. Create `MyScaleSettings` object with parameters


    ```python
    config = MyScaleSetting(host="<your-backend-url>", port=8443, ...)
    index = MyScale(embedding_function, config)
    index.add_documents(...)
    ```
"""
logger.info("There are two ways to set up parameters for myscale index.")



loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

for d in docs:
    d.metadata = {"some": "metadata"}
docsearch = MyScale.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

logger.debug(docs[0].page_content)

"""
## Get connection info and data schema
"""
logger.info("## Get connection info and data schema")

logger.debug(str(docsearch))

"""
## Filtering

You can have direct access to myscale SQL where statement. You can write `WHERE` clause following standard SQL.

**NOTE**: Please be aware of SQL injection, this interface must not be directly called by end-user.

If you customized your `column_map` under your setting, you search with filter like this:
"""
logger.info("## Filtering")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}

docsearch = MyScale.from_documents(docs, embeddings)

"""
### Similarity search with score

The returned distance score is cosine distance. Therefore, a lower score is better.
"""
logger.info("### Similarity search with score")

meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    logger.debug(dist, d.metadata, d.page_content[:20] + "...")

"""
## Deleting your data

You can either drop the table with `.drop()` method or partially delete your data with `.delete()` method.
"""
logger.info("## Deleting your data")

docsearch.delete(where_str=f"{docsearch.metadata_column}.doc_id < 5")
meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    logger.debug(dist, d.metadata, d.page_content[:20] + "...")

docsearch.drop()

logger.info("\n\n[DONE]", bright=True)