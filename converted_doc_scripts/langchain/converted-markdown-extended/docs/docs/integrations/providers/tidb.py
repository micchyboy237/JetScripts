from jet.logger import logger
from langchain_community.chat_message_histories import TiDBChatMessageHistory
from langchain_community.document_loaders import TiDBLoader
from langchain_community.vectorstores import TiDBVectorStore
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

> [TiDB Cloud](https://www.pingcap.com/tidb-serverless), is a comprehensive Database-as-a-Service (DBaaS) solution,
> that provides dedicated and serverless options. `TiDB Serverless` is now integrating
> a built-in vector search into the MySQL landscape. With this enhancement, you can seamlessly
> develop AI applications using `TiDB Serverless` without the need for a new database or additional
> technical stacks. Create a free TiDB Serverless cluster and start using the vector search feature at https://pingcap.com/ai.


## Installation and Setup

You have to get the connection details for the TiDB database.
Visit the [TiDB Cloud](https://tidbcloud.com/) to get the connection details.
"""
logger.info("# TiDB")



"""

Please refer the details [here](/docs/integrations/document_loaders/tidb).

## Vector store


Please refer the details [here](/docs/integrations/vectorstores/tidb_vector).


## Memory

"""
logger.info("## Vector store")

logger.info("\n\n[DONE]", bright=True)