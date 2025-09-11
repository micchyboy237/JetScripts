from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from jet.logger import logger
from langchain_community.chat_message_histories import (
CassandraChatMessageHistory,
)
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
# Cassandra 

>[Apache Cassandra®](https://cassandra.apache.org) is a `NoSQL`, row-oriented, highly scalable and highly available database, well suited for storing large amounts of data.

>`Cassandra` is a good choice for storing chat message history because it is easy to scale and can handle a large number of writes.

This notebook goes over how to use Cassandra to store chat message history.

## Setting up

To run this notebook you need either a running `Cassandra` cluster or a `DataStax Astra DB` instance running in the cloud (you can get one for free at [datastax.com](https://astra.datastax.com)). Check [cassio.org](https://cassio.org/start_here/) for more information.
"""
logger.info("# Cassandra")

# %pip install --upgrade --quiet  "cassio>=0.1.0 langchain-community"

"""
### Set up the database connection parameters and secrets
"""
logger.info("### Set up the database connection parameters and secrets")

# import getpass

database_mode = (input("\n(C)assandra or (A)stra DB? ")).upper()

keyspace_name = input("\nKeyspace name? ")

if database_mode == "A":
#     ASTRA_DB_APPLICATION_TOKEN = getpass.getpass('\nAstra DB Token ("AstraCS:...") ')
    ASTRA_DB_SECURE_BUNDLE_PATH = input("Full path to your Secure Connect Bundle? ")
elif database_mode == "C":
    CASSANDRA_CONTACT_POINTS = input(
        "Contact points? (comma-separated, empty for localhost) "
    ).strip()

"""
Depending on whether local or cloud-based Astra DB, create the corresponding database connection "Session" object.
"""
logger.info("Depending on whether local or cloud-based Astra DB, create the corresponding database connection "Session" object.")


if database_mode == "C":
    if CASSANDRA_CONTACT_POINTS:
        cluster = Cluster(
            [cp.strip() for cp in CASSANDRA_CONTACT_POINTS.split(",") if cp.strip()]
        )
    else:
        cluster = Cluster()
    session = cluster.connect()
elif database_mode == "A":
    ASTRA_DB_CLIENT_ID = "token"
    cluster = Cluster(
        cloud={
            "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
        },
        auth_provider=PlainTextAuthProvider(
            ASTRA_DB_CLIENT_ID,
            ASTRA_DB_APPLICATION_TOKEN,
        ),
    )
    session = cluster.connect()
else:
    raise NotImplementedError

"""
## Example
"""
logger.info("## Example")


message_history = CassandraChatMessageHistory(
    session_id="test-session",
    session=session,
    keyspace=keyspace_name,
)

message_history.add_user_message("hi!")

message_history.add_ai_message("whats up?")

message_history.messages

"""
#### Attribution statement

> Apache Cassandra, Cassandra and Apache are either registered trademarks or trademarks of the [Apache Software Foundation](http://www.apache.org/) in the United States and/or other countries.
"""
logger.info("#### Attribution statement")


logger.info("\n\n[DONE]", bright=True)