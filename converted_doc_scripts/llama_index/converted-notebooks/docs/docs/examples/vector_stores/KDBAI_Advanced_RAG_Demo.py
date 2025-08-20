from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import datetime
import kdbai_client as kdbai
import os
import pandas as pd
import re
import shutil
import time
import urllib


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Advanced RAG with temporal filters using LlamaIndex and KDB.AI vector store

##### Note: This example requires a KDB.AI endpoint and API key. Sign up for a free [KDB.AI account](https://kdb.ai/get-started).

> [KDB.AI](https://kdb.ai/) is a powerful knowledge-based vector database and search engine that allows you to build scalable, reliable AI applications, using real-time data, by providing advanced search, recommendation and personalization.

This example demonstrates how to use KDB.AI to run semantic search, summarization and analysis of financial regulations around some specific moment in time.

To access your end point and API keys, sign up to KDB.AI here.

To set up your development environment, follow the instructions on the KDB.AI pre-requisites page.

The following examples demonstrate some of the ways you can interact with KDB.AI through LlamaIndex.

## Install dependencies with Pip

In order to successfully run this sample, note the following steps depending on where you are running this notebook:

-***Run Locally / Private Environment:*** The [Setup](https://github.com/KxSystems/kdbai-samples/blob/main/README.md#setup) steps in the repository's `README.md` will guide you on prerequisites and how to run this with Jupyter.


-***Colab / Hosted Environment:*** Open this notebook in Colab and run through the cells.
"""
logger.info(
    "# Advanced RAG with temporal filters using LlamaIndex and KDB.AI vector store")

# !pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-readers-file llama-index-vector-stores-kdbai
# !pip install kdbai_client pandas

"""
## Import dependencies
"""
logger.info("## Import dependencies")

# from getpass import getpass


OUTDIR = "pdf"
RESET = True

"""
#### Set MLX API key and choose the LLM and Embedding model to use:
"""
logger.info("#### Set MLX API key and choose the LLM and Embedding model to use:")

# os.environ["OPENAI_API_KEY"] = (
#     os.environ["OPENAI_API_KEY"]
#     if "OPENAI_API_KEY" in os.environ
#     else getpass("MLX API Key: ")
)

    # from getpass import getpass

    # if "OPENAI_API_KEY" in os.environ:
    #     KDBAI_API_KEY = os.environ["OPENAI_API_KEY"]
    else:
    #     OPENAI_API_KEY = getpass("OPENAI API KEY: ")
    #     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    EMBEDDING_MODEL= "mxbai-embed-large"
    GENERATION_MODEL= "qwen3-1.7b-4bit"

    llm= MLXLlamaIndexLLMAdapter(model=GENERATION_MODEL)
    embed_model= MLXEmbedding(model=EMBEDDING_MODEL)

    Settings.llm= llm
    Settings.embed_model= embed_model

    """
## Create KDB.AI session and table
"""
    logger.info("## Create KDB.AI session and table")

    # from getpass import getpass

    """
##### Option 1. KDB.AI Cloud

To use KDB.AI Cloud, you will need two session details - a URL endpoint and an API key.
To get these you can sign up for free [here](https://trykdb.kx.com/kdbai/signup).

You can connect to a KDB.AI Cloud session using `kdbai.Session` and passing the session URL endpoint and API key details from your KDB.AI Cloud portal.

If the environment variables `KDBAI_ENDPOINTS` and `KDBAI_API_KEY` exist on your system containing your KDB.AI Cloud portal details, these variables will automatically be used to connect.
If these do not exist, it will prompt you to enter your KDB.AI Cloud portal session URL endpoint and API key details.
"""
    logger.info("##### Option 1. KDB.AI Cloud")

    KDBAI_ENDPOINT= (
os.environ["KDBAI_ENDPOINT"]
if "KDBAI_ENDPOINT" in os.environ
else input("KDB.AI endpoint: ")
)
    KDBAI_API_KEY= (
os.environ["KDBAI_API_KEY"]
if "KDBAI_API_KEY" in os.environ
#     else getpass("KDB.AI API key: ")
)

    session= kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)

    """
##### Option 2. KDB.AI Server

To use KDB.AI Server, you will need download and run your own container.
To do this, you will first need to sign up for free [here](https://trykdb.kx.com/kdbaiserver/signup/).

You will receive an email with the required license file and bearer token needed to download your instance.
Follow instructions in the signup email to get your session up and running.

Once the [setup steps](https://code.kx.com/kdbai/gettingStarted/kdb-ai-server-setup.html) are complete you can then connect to your KDB.AI Server session using `kdbai.Session` and passing your local endpoint.
"""
    logger.info("##### Option 2. KDB.AI Server")



    """
### Create the schema for your KDB.AI table

***!!! Note:*** The 'dims' parameter in the embedding column must reflect the output dimensions of the embedding model you choose.


- MLX 'mxbai-embed-large' outputs 1536 dimensions.
"""
    logger.info("### Create the schema for your KDB.AI table")

    schema= [
{"name": "document_id", "type": "bytes"},
{"name": "text", "type": "bytes"},
{"name": "embeddings", "type": "float32s"},
{"name": "title", "type": "str"},
{"name": "publication_date", "type": "datetime64[ns]"},
]


    indexFlat= {
"name": "flat_index",
"type": "flat",
"column": "embeddings",
"params": {"dims": 1536, "metric": "L2"},
}

    KDBAI_TABLE_NAME = "reports"
    database = session.database("default")

    for table in database.tables:
    if table.name == KDBAI_TABLE_NAME:
        table.drop()
        break

    table = database.create_table(
KDBAI_TABLE_NAME, schema = schema, indexes = [indexFlat]
)

    """
## Financial reports urls and metadata
"""
    logger.info("## Financial reports urls and metadata")

    INPUT_URLS= [
"https://www.govinfo.gov/content/pkg/PLAW-106publ102/pdf/PLAW-106publ102.pdf",
"https://www.govinfo.gov/content/pkg/PLAW-111publ203/pdf/PLAW-111publ203.pdf",
]

    METADATA= {
"pdf/PLAW-106publ102.pdf": {
    "title": "GRAMM–LEACH–BLILEY ACT, 1999",
    "publication_date": pd.to_datetime("1999-11-12"),
},
"pdf/PLAW-111publ203.pdf": {
    "title": "DODD-FRANK WALL STREET REFORM AND CONSUMER PROTECTION ACT, 2010",
    "publication_date": pd.to_datetime("2010-07-21"),
},
}

    """
## Download PDF files locally
"""
    logger.info("## Download PDF files locally")

    # %%time

    CHUNK_SIZE = 512 * 1024


    def download_file(url):
    logger.debug("Downloading %s..." % url)
    out = os.path.join(OUTDIR, os.path.basename(url))
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        logging.exception("Failed to download %s !" % url)
    else:
        with open(out, "wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if chunk:
                    f.write(chunk)
                else:
                    break
    return out


    if RESET:
    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.mkdir(OUTDIR)

    local_files = [download_file(x) for x in INPUT_URLS]
    local_files[:10]

    """
## Load local PDF files with LlamaIndex
"""
    logger.info("## Load local PDF files with LlamaIndex")

    # %%time


    def get_metadata(filepath):
    return METADATA[filepath]


    documents = SimpleDirectoryReader(
input_files = local_files,
file_metadata = get_metadata,
)

    docs= documents.load_data()
    len(docs)

    """
## Setup LlamaIndex RAG pipeline using KDB.AI vector store
"""
    logger.info("## Setup LlamaIndex RAG pipeline using KDB.AI vector store")

    # %%time

    vector_store= KDBAIVectorStore(table)

    storage_context= StorageContext.from_defaults(vector_store=vector_store)
    index= VectorStoreIndex.from_documents(
docs,
storage_context = storage_context,
transformations = [SentenceSplitter(chunk_size=2048, chunk_overlap=0)],
)

    table.query()

    """
## Setup the LlamaIndex Query Engine
"""
    logger.info("## Setup the LlamaIndex Query Engine")

    # %%time

    K= 15

    query_engine= index.as_query_engine(
similarity_top_k = K,
vector_store_kwargs = {
    "index": "flat_index",
    "filter": [["<", "publication_date", datetime.date(2008, 9, 15)]],
    "sort_columns": "publication_date",
},
)

    """
## Before the 2008 crisis
"""
    logger.info("## Before the 2008 crisis")

    # %%time

    result= query_engine.query(
"""
    What was the main financial regulation in the US before the 2008 financial crisis ?
    """
)
    logger.debug(result.response)

    # %%time

    result= query_engine.query(
"""
    Is the Gramm-Leach-Bliley Act of 1999 enough to prevent the 2008 crisis. Search the document and explain its strenghts and weaknesses to regulate the US stock market.
    """
)
    logger.debug(result.response)

    """
## After the 2008 crisis
"""
    logger.info("## After the 2008 crisis")

    # %%time

    K= 15

    query_engine= index.as_query_engine(
similarity_top_k = K,
vector_store_kwargs = {
    "index": "flat_index",
    "filter": [[">=", "publication_date", datetime.date(2008, 9, 15)]],
    "sort_columns": "publication_date",
},
)

    # %%time

    result= query_engine.query(
"""
    What happened on the 15th of September 2008 ?
    """
)
    logger.debug(result.response)

    # %%time

    result= query_engine.query(
"""
    What was the new US financial regulation enacted after the 2008 crisis to increase the market regulation and to improve consumer sentiment ?
    """
)
    logger.debug(result.response)

    """
## In depth analysis
"""
    logger.info("## In depth analysis")

    # %%time

    K= 20

    query_engine= index.as_query_engine(
similarity_top_k = K,
vector_store_kwargs = {
    "index": "flat_index",
    "sort_columns": "publication_date",
},
)

    # %%time

    result= query_engine.query(
"""
    Analyse the US financial regulations before and after the 2008 crisis and produce a report of all related arguments to explain what happened, and to ensure that does not happen again.
    Use both the provided context and your own knowledge but do mention explicitely which one you use.
    """
)
    logger.debug(result.response)

    """
## Delete the KDB.AI Table

Once finished with the table, it is best practice to drop it.
"""
    logger.info("## Delete the KDB.AI Table")

    table.drop()

    """
#### Take Our Survey
We hope you found this sample helpful! Your feedback is important to us, and we would appreciate it if you could take a moment to fill out our brief survey. Your input helps us improve our content.

Take the [Survey](https://delighted.com/t/kWYXv316)
"""
    logger.info("#### Take Our Survey")

    logger.info("\n\n[DONE]", bright=True)
