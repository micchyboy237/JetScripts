from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import AnalyticDB
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
# AnalyticDB

>[AnalyticDB for PostgreSQL](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) is a massively parallel processing (MPP) data warehousing service that is designed to analyze large volumes of data online.

>`AnalyticDB for PostgreSQL` is developed based on the open-source `Greenplum Database` project and is enhanced with in-depth extensions by `Alibaba Cloud`. AnalyticDB for PostgreSQL is compatible with the ANSI SQL 2003 syntax and the PostgreSQL and Oracle database ecosystems. AnalyticDB for PostgreSQL also supports row store and column store. AnalyticDB for PostgreSQL processes petabytes of data offline at a high performance level and supports highly concurrent online queries.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `AnalyticDB` vector database.
To run, you should have an [AnalyticDB](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) instance up and running:

- Using [AnalyticDB Cloud Vector Database](https://www.alibabacloud.com/product/hybriddb-postgresql). Click here to fast deploy it.
"""
logger.info("# AnalyticDB")


"""
Split documents and get embeddings by call Ollama API
"""
logger.info("Split documents and get embeddings by call Ollama API")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
Connect to AnalyticDB by setting related ENVIRONMENTS.
```
export PG_HOST={your_analyticdb_hostname}
export PG_PORT={your_analyticdb_port} # Optional, default is 5432
export PG_DATABASE={your_database} # Optional, default is postgres
export PG_USER={database_username}
export PG_PASSWORD={database_password}
```

Then store your embeddings and documents into AnalyticDB
"""
logger.info("Connect to AnalyticDB by setting related ENVIRONMENTS.")


connection_string = AnalyticDB.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD", "postgres"),
)

vector_db = AnalyticDB.from_documents(
    docs,
    embeddings,
    connection_string=connection_string,
)

"""
Query and retrieve data
"""
logger.info("Query and retrieve data")

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)

logger.debug(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)
