from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import Relyt
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
# Relyt

>[Relyt](https://docs.relyt.cn/docs/vector-engine/use/) is a cloud native data warehousing service that is designed to analyze large volumes of data online.

>`Relyt` is compatible with the ANSI SQL 2003 syntax and the PostgreSQL and Oracle database ecosystems. Relyt also supports row store and column store. Relyt processes petabytes of data offline at a high performance level and supports highly concurrent online queries.

This notebook shows how to use functionality related to the `Relyt` vector database.
To run, you should have an [Relyt](https://docs.relyt.cn/) instance up and running:
- Using [Relyt Vector Database](https://docs.relyt.cn/docs/vector-engine/use/). Click here to fast deploy it.
"""
logger.info("# Relyt")

# %pip install "pgvecto_rs[sdk]" langchain-community


"""
Split documents and get embeddings by call community API
"""
logger.info("Split documents and get embeddings by call community API")

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = FakeEmbeddings(size=1536)

"""
Connect to Relyt by setting related ENVIRONMENTS.
```
export PG_HOST={your_relyt_hostname}
export PG_PORT={your_relyt_port} # Optional, default is 5432
export PG_DATABASE={your_database} # Optional, default is postgres
export PG_USER={database_username}
export PG_PASSWORD={database_password}
```

Then store your embeddings and documents into Relyt
"""
logger.info("Connect to Relyt by setting related ENVIRONMENTS.")


connection_string = Relyt.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD", "postgres"),
)

vector_db = Relyt.from_documents(
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