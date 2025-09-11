from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Hologres
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
# Hologres

>[Hologres](https://www.alibabacloud.com/help/en/hologres/latest/introduction) is a unified real-time data warehousing service developed by Alibaba Cloud. You can use Hologres to write, update, process, and analyze large amounts of data in real time. 
>Hologres supports standard SQL syntax, is compatible with PostgreSQL, and supports most PostgreSQL functions. Hologres supports online analytical processing (OLAP) and ad hoc analysis for up to petabytes of data, and provides high-concurrency and low-latency online data services. 

>Hologres provides **vector database** functionality by adopting [Proxima](https://www.alibabacloud.com/help/en/hologres/latest/vector-processing).
>Proxima is a high-performance software library developed by Alibaba DAMO Academy. It allows you to search for the nearest neighbors of vectors. Proxima provides higher stability and performance than similar open-source software such as Faiss. Proxima allows you to search for similar text or image embeddings with high throughput and low latency. Hologres is deeply integrated with Proxima to provide a high-performance vector search service.

This notebook shows how to use functionality related to the `Hologres Proxima` vector database.
Click [here](https://www.alibabacloud.com/zh/product/hologres) to fast deploy a Hologres cloud instance.
"""
logger.info("# Hologres")

# %pip install --upgrade --quiet  langchain_community hologres-vector


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
Connect to Hologres by setting related ENVIRONMENTS.
```
export PG_HOST={host}
export PG_PORT={port} # Optional, default is 80
export PG_DATABASE={db_name} # Optional, default is postgres
export PG_USER={username}
export PG_PASSWORD={password}
```

Then store your embeddings and documents into Hologres
"""
logger.info("Connect to Hologres by setting related ENVIRONMENTS.")


connection_string = Hologres.connection_string_from_db_params(
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", "80")),
    database=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "postgres"),
)

vector_db = Hologres.from_documents(
    docs,
    embeddings,
    connection_string=connection_string,
    table_name="langchain_example_embeddings",
)

"""
Query and retrieve data
"""
logger.info("Query and retrieve data")

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)

logger.debug(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)