from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.pgvecto_rs import PGVectoRsStore
from pgvecto_rs.sdk import PGVectoRs
import logging
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
# pgvecto.rs

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/PGVectoRsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Firstly, you will probably need to install dependencies :
"""
logger.info("# pgvecto.rs")

# %pip install llama-index-vector-stores-pgvecto-rs

# %pip install llama-index "pgvecto_rs[sdk]"

"""
Then start the pgvecto.rs server as the [official document suggests](https://github.com/tensorchord/pgvecto.rs#installation):
"""
logger.info("Then start the pgvecto.rs server as the [official document suggests](https://github.com/tensorchord/pgvecto.rs#installation):")

# !docker run --name pgvecto-rs-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d tensorchord/pgvecto-rs:latest

"""
Setup the logger.
"""
logger.info("Setup the logger.")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
#### Creating a pgvecto_rs client
"""
logger.info("#### Creating a pgvecto_rs client")


URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
    port=os.getenv("DB_PORT", "5432"),
    host=os.getenv("DB_HOST", "localhost"),
    username=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "mysecretpassword"),
    db_name=os.getenv("DB_NAME", "postgres"),
)

client = PGVectoRs(
    db_url=URL,
    collection_name="example",
    dimension=1536,  # Using OllamaFunctionCalling’s text-embedding-ada-002
)

"""
#### Setup OllamaFunctionCalling
"""
logger.info("#### Setup OllamaFunctionCalling")


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
#### Load documents, build the PGVectoRsStore and VectorStoreIndex
"""
logger.info("#### Load documents, build the PGVectoRsStore and VectorStoreIndex")



"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


vector_store = PGVectoRsStore(client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)