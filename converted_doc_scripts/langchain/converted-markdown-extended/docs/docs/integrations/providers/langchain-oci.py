from jet.logger import logger
from langchain_community.document_loaders.oracleai import OracleDocLoader
from langchain_community.document_loaders.oracleai import OracleTextSplitter
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.utilities.oracleai import OracleSummary
from langchain_community.vectorstores.oraclevs import OracleVS
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
# OracleAI Vector Search

Oracle AI Vector Search is designed for Artificial Intelligence (AI) workloads that allows you to query data based on semantics, rather than keywords.
One of the biggest benefits of Oracle AI Vector Search is that semantic search on unstructured data can be combined with relational search on business data in one single system.
This is not only powerful but also significantly more effective because you don't need to add a specialized vector database, eliminating the pain of data fragmentation between multiple systems.

In addition, your vectors can benefit from all of Oracle Database’s most powerful features, like the following:

 * [Partitioning Support](https://www.oracle.com/database/technologies/partitioning.html)
 * [Real Application Clusters scalability](https://www.oracle.com/database/real-application-clusters/)
 * [Exadata smart scans](https://www.oracle.com/database/technologies/exadata/software/smartscan/)
 * [Shard processing across geographically distributed databases](https://www.oracle.com/database/distributed-database/)
 * [Transactions](https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/transactions.html)
 * [Parallel SQL](https://docs.oracle.com/en/database/oracle/oracle-database/21/vldbg/parallel-exec-intro.html#GUID-D28717E4-0F77-44F5-BB4E-234C31D4E4BA)
 * [Disaster recovery](https://www.oracle.com/database/data-guard/)
 * [Security](https://www.oracle.com/security/database-security/)
 * [Oracle Machine Learning](https://www.oracle.com/artificial-intelligence/database-machine-learning/)
 * [Oracle Graph Database](https://www.oracle.com/database/integrated-graph-database/)
 * [Oracle Spatial and Graph](https://www.oracle.com/database/spatial/)
 * [Oracle Blockchain](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_blockchain_table.html#GUID-B469E277-978E-4378-A8C1-26D3FF96C9A6)
 * [JSON](https://docs.oracle.com/en/database/oracle/oracle-database/23/adjsn/json-in-oracle-database.html)


## Document Loaders

Please check the [usage example](/docs/integrations/document_loaders/oracleai).
"""
logger.info("# OracleAI Vector Search")


"""
## Text Splitter

Please check the [usage example](/docs/integrations/document_loaders/oracleai).
"""
logger.info("## Text Splitter")


"""
## Embeddings

Please check the [usage example](/docs/integrations/text_embedding/oracleai).
"""
logger.info("## Embeddings")


"""
## Summary

Please check the [usage example](/docs/integrations/tools/oracleai).
"""
logger.info("## Summary")


"""
## Vector Store

Please check the [usage example](/docs/integrations/vectorstores/oracle).
"""
logger.info("## Vector Store")


"""
## End to End Demo

Please check the [Oracle AI Vector Search End-to-End Demo Guide](https://github.com/langchain-ai/langchain/blob/master/cookbook/oracleai_demo.ipynb).
"""
logger.info("## End to End Demo")

logger.info("\n\n[DONE]", bright=True)