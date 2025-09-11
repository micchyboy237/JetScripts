from jet.logger import logger
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
# OceanBase

[OceanBase Database](https://github.com/oceanbase/oceanbase) is a distributed relational database.
It is developed entirely by Ant Group. The OceanBase Database is built on a common server cluster.
Based on the Paxos protocol and its distributed structure, the OceanBase Database provides high availability and linear scalability.

OceanBase currently has the ability to store vectors. Users can easily perform the following operations with SQL:

- Create a table containing vector type fields;
- Create a vector index table based on the HNSW algorithm;
- Perform vector approximate nearest neighbor queries;
- ...

## Installation
"""
logger.info("# OceanBase")

pip install -U langchain-oceanbase

"""
We recommend using Docker to deploy OceanBase:
"""
logger.info("We recommend using Docker to deploy OceanBase:")

docker run --name=ob433 -e MODE=slim -p 2881:2881 -d oceanbase/oceanbase-ce:4.3.3.0-100000132024100711

"""
[More methods to deploy OceanBase cluster](https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md)

### Usage

For a more detailed walkthrough of the OceanBase Wrapper, see [this notebook](https://github.com/oceanbase/langchain-oceanbase/blob/main/docs/vectorstores.ipynb)
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)