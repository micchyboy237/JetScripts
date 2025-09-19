from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.tidb import TiDBGraphStore
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
import logging
import openai
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# TiDB Graph Store
"""
logger.info("# TiDB Graph Store")

# %pip install llama-index-llms-ollama
# %pip install llama-index-graph-stores-tidb
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-azure-openai


# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

llm = OllamaFunctionCalling(temperature=0, model="llama3.2")
Settings.llm = llm
Settings.chunk_size = 512


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://<foo-bar>.openai.azure.com"
openai.api_version = "2022-12-01"
# os.environ["OPENAI_API_KEY"] = "<your-openai-key>"
# openai.api_key = os.getenv("OPENAI_API_KEY")

llm = AzureOllamaFunctionCallingAdapter(
    deployment_name="<foo-bar-deployment>",
    temperature=0,
    openai_api_version=openai.api_version,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)

embedding_llm = HuggingFaceEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 512

"""
## Using Knowledge Graph with TiDB

### Prepare a TiDB cluster

- [TiDB Cloud](https://tidb.cloud/) [Recommended], a fully managed TiDB service that frees you from the complexity of database operations.
- [TiUP](https://docs.pingcap.com/tidb/stable/tiup-overview), use `tiup playground`` to create a local TiDB cluster for testing.

#### Get TiDB connection string

For example: `mysql+pymysql://user:password@host:4000/dbname`, in TiDBGraphStore we use pymysql as the db driver, so the connection string should be `mysql+pymysql://...`.

If you are using a TiDB Cloud serverless cluster with public endpoint, it requires TLS connection, so the connection string should be like `mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true`.

Replace `user`, `password`, `host`, `dbname` with your own values.

### Initialize TiDBGraphStore
"""
logger.info("## Using Knowledge Graph with TiDB")


graph_store = TiDBGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname"
)

"""
### Instantiate TiDB KG Indexes
"""
logger.info("### Instantiate TiDB KG Indexes")


documents = SimpleDirectoryReader(
    "../../../examples/data/paul_graham/"
).load_data()

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)

"""
#### Querying the Knowledge Graph
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)


display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)
