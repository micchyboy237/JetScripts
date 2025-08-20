from __future__ import absolute_import
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.database import DatabaseReader
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/DatabaseReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Database Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Database Reader")

# %pip install llama-index-readers-database

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# os.environ["OPENAI_API_KEY"] = ""


db = DatabaseReader(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="FakeExamplePassword",  # Database Password
    dbname="postgres",  # Database Name
)

logger.debug(type(db))
logger.debug(type(db.load_data))

logger.debug(type(db.sql_database))
logger.debug(type(db.sql_database.from_uri))
logger.debug(type(db.sql_database.get_single_table_info))
logger.debug(type(db.sql_database.get_table_columns))
logger.debug(type(db.sql_database.get_usable_table_names))
logger.debug(type(db.sql_database.insert_into_table))
logger.debug(type(db.sql_database.run_sql))
logger.debug(type(db.sql_database.dialect))
logger.debug(type(db.sql_database.engine))

logger.debug(type(db.sql_database))
db_from_sql_database = DatabaseReader(sql_database=db.sql_database)
logger.debug(type(db_from_sql_database))

logger.debug(type(db.sql_database.engine))
db_from_engine = DatabaseReader(engine=db.sql_database.engine)
logger.debug(type(db_from_engine))

logger.debug(type(db.uri))
db_from_uri = DatabaseReader(uri=db.uri)
logger.debug(type(db_from_uri))

query = f"""
    SELECT
        CONCAT(name, ' is ', age, ' years old.') AS text
    FROM public.users
    WHERE age >= 18
    """

texts = db.sql_database.run_sql(command=query)

logger.debug(type(texts))

logger.debug(texts)

documents = db.load_data(query=query)

logger.debug(type(documents))

logger.debug(documents)

index = VectorStoreIndex.from_documents(documents)

logger.info("\n\n[DONE]", bright=True)