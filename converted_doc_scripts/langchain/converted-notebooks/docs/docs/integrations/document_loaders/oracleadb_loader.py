from jet.logger import logger
from langchain_community.document_loaders import OracleAutonomousDatabaseLoader
from settings import s
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
# Oracle Autonomous Database

Oracle Autonomous Database is a cloud database that uses machine learning to automate database tuning, security, backups, updates, and other routine management tasks traditionally performed by DBAs.

This notebook covers how to load documents from Oracle Autonomous Database.

## Prerequisites
1. Install python-oracledb:

   `pip install oracledb`
  
   See [Installing python-oracledb](https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html).

2. A database that python-oracledb's default 'Thin' mode can connected to. This is true of Oracle Autonomous Database, see [python-oracledb Architecture](https://python-oracledb.readthedocs.io/en/latest/user_guide/introduction.html#architecture).

## Instructions
"""
logger.info("# Oracle Autonomous Database")

pip install oracledb


"""
With mutual TLS authentication (mTLS), wallet_location and wallet_password parameters are required to create the connection. See python-oracledb documentation [Connecting to Oracle Cloud Autonomous Databases](https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#connecting-to-oracle-cloud-autonomous-databases).
"""
logger.info("With mutual TLS authentication (mTLS), wallet_location and wallet_password parameters are required to create the connection. See python-oracledb documentation [Connecting to Oracle Cloud Autonomous Databases](https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html#connecting-to-oracle-cloud-autonomous-databases).")

SQL_QUERY = "select prod_id, time_id from sh.costs fetch first 5 rows only"

doc_loader_1 = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    config_dir=s.CONFIG_DIR,
    wallet_location=s.WALLET_LOCATION,
    wallet_password=s.PASSWORD,
    tns_name=s.TNS_NAME,
)
doc_1 = doc_loader_1.load()

doc_loader_2 = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    connection_string=s.CONNECTION_STRING,
    wallet_location=s.WALLET_LOCATION,
    wallet_password=s.PASSWORD,
)
doc_2 = doc_loader_2.load()

"""
With 1-way TLS authentication, only the database credentials and connection string are required to establish a connection.
The example below also shows passing bind variable values with the argument "parameters".
"""
logger.info("With 1-way TLS authentication, only the database credentials and connection string are required to establish a connection.")

SQL_QUERY = "select channel_id, channel_desc from sh.channels where channel_desc = :1 fetch first 5 rows only"

doc_loader_3 = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    config_dir=s.CONFIG_DIR,
    tns_name=s.TNS_NAME,
    parameters=["Direct Sales"],
)
doc_3 = doc_loader_3.load()

doc_loader_4 = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    connection_string=s.CONNECTION_STRING,
    parameters=["Direct Sales"],
)
doc_4 = doc_loader_4.load()

logger.info("\n\n[DONE]", bright=True)