from jet.logger import logger
from langchain_community.document_loaders import TiDBLoader
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
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
# TiDB

> [TiDB Cloud](https://tidbcloud.com/), is a comprehensive Database-as-a-Service (DBaaS) solution, that provides dedicated and serverless options. TiDB Serverless is now integrating a built-in vector search into the MySQL landscape. With this enhancement, you can seamlessly develop AI applications using TiDB Serverless without the need for a new database or additional technical stacks. Be among the first to experience it by joining the waitlist for the private beta at https://tidb.cloud/ai.

This notebook introduces how to use `TiDBLoader` to load data from TiDB in langchain.

## Prerequisites

Before using the `TiDBLoader`, we will install the following dependencies:
"""
logger.info("# TiDB")

# %pip install --upgrade --quiet langchain

"""
Then, we will configure the connection to a TiDB. In this notebook, we will follow the standard connection method provided by TiDB Cloud to establish a secure and efficient database connection.
"""
logger.info("Then, we will configure the connection to a TiDB. In this notebook, we will follow the standard connection method provided by TiDB Cloud to establish a secure and efficient database connection.")

# import getpass

tidb_connection_string_template = "mysql+pymysql://<USER>:<PASSWORD>@<HOST>:4000/<DB>?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
# tidb_password = getpass.getpass("Input your TiDB password:")
tidb_connection_string = tidb_connection_string_template.replace(
    "<PASSWORD>", tidb_password
)

"""
## Load Data from TiDB

Here's a breakdown of some key arguments you can use to customize the behavior of the `TiDBLoader`:

- `query` (str): This is the SQL query to be executed against the TiDB database. The query should select the data you want to load into your `Document` objects. 
    For instance, you might use a query like `"SELECT * FROM my_table"` to fetch all data from `my_table`.

- `page_content_columns` (Optional[List[str]]): Specifies the list of column names whose values should be included in the `page_content` of each `Document` object. 
    If set to `None` (the default), all columns returned by the query are included in `page_content`. This allows you to tailor the content of each document based on specific columns of your data.

- `metadata_columns` (Optional[List[str]]): Specifies the list of column names whose values should be included in the `metadata` of each `Document` object. 
    By default, this list is empty, meaning no metadata will be included unless explicitly specified. This is useful for including additional information about each document that doesn't form part of the main content but is still valuable for processing or analysis.
"""
logger.info("## Load Data from TiDB")


engine = create_engine(tidb_connection_string)
metadata = MetaData()
table_name = "test_tidb_loader"

test_table = Table(
    table_name,
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255)),
    Column("description", String(255)),
)
metadata.create_all(engine)


with engine.connect() as connection:
    transaction = connection.begin()
    try:
        connection.execute(
            test_table.insert(),
            [
                {"name": "Item 1", "description": "Description of Item 1"},
                {"name": "Item 2", "description": "Description of Item 2"},
                {"name": "Item 3", "description": "Description of Item 3"},
            ],
        )
        transaction.commit()
    except:
        transaction.rollback()
        raise


loader = TiDBLoader(
    connection_string=tidb_connection_string,
    query=f"SELECT * FROM {table_name};",
    page_content_columns=["name", "description"],
    metadata_columns=["id"],
)

documents = loader.load()

for doc in documents:
    logger.debug("-" * 30)
    logger.debug(f"content: {doc.page_content}\nmetada: {doc.metadata}")

test_table.drop(bind=engine)

logger.info("\n\n[DONE]", bright=True)