from jet.logger import logger
from langchain_community.document_loaders import MaxComputeLoader
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
# Alibaba Cloud MaxCompute

>[Alibaba Cloud MaxCompute](https://www.alibabacloud.com/product/maxcompute) (previously known as ODPS) is a general purpose, fully managed, multi-tenancy data processing platform for large-scale data warehousing. MaxCompute supports various data importing solutions and distributed computing models, enabling users to effectively query massive datasets, reduce production costs, and ensure data security.

The `MaxComputeLoader` lets you execute a MaxCompute SQL query and loads the results as one document per row.
"""
logger.info("# Alibaba Cloud MaxCompute")

# %pip install --upgrade --quiet  pyodps

"""
## Basic Usage
To instantiate the loader you'll need a SQL query to execute, your MaxCompute endpoint and project name, and your access ID and secret access key. The access ID and secret access key can either be passed in direct via the `access_id` and `secret_access_key` parameters or they can be set as environment variables `MAX_COMPUTE_ACCESS_ID` and `MAX_COMPUTE_SECRET_ACCESS_KEY`.
"""
logger.info("## Basic Usage")


base_query = """
SELECT *
FROM (
    SELECT 1 AS id, 'content1' AS content, 'meta_info1' AS meta_info
    UNION ALL
    SELECT 2 AS id, 'content2' AS content, 'meta_info2' AS meta_info
    UNION ALL
    SELECT 3 AS id, 'content3' AS content, 'meta_info3' AS meta_info
) mydata;
"""

endpoint = "<ENDPOINT>"
project = "<PROJECT>"
ACCESS_ID = "<ACCESS ID>"
SECRET_ACCESS_KEY = "<SECRET ACCESS KEY>"

loader = MaxComputeLoader.from_params(
    base_query,
    endpoint,
    project,
    access_id=ACCESS_ID,
    secret_access_key=SECRET_ACCESS_KEY,
)
data = loader.load()

logger.debug(data)

logger.debug(data[0].page_content)

logger.debug(data[0].metadata)

"""
## Specifying Which Columns are Content vs Metadata
You can configure which subset of columns should be loaded as the contents of the Document and which as the metadata using the `page_content_columns` and `metadata_columns` parameters.
"""
logger.info("## Specifying Which Columns are Content vs Metadata")

loader = MaxComputeLoader.from_params(
    base_query,
    endpoint,
    project,
    page_content_columns=["content"],  # Specify Document page content
    metadata_columns=["id", "meta_info"],  # Specify Document metadata
    access_id=ACCESS_ID,
    secret_access_key=SECRET_ACCESS_KEY,
)
data = loader.load()

logger.debug(data[0].page_content)

logger.debug(data[0].metadata)

logger.info("\n\n[DONE]", bright=True)