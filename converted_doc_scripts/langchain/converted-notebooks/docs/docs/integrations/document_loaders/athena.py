from jet.logger import logger
from langchain_community.document_loaders.athena import AthenaLoader
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
# Athena

>[Amazon Athena](https://aws.amazon.com/athena/) is a serverless, interactive analytics service built
>on open-source frameworks, supporting open-table and file formats. `Athena` provides a simplified,
>flexible way to analyze petabytes of data where it lives. Analyze data or build applications
>from an Amazon Simple Storage Service (S3) data lake and 30 data sources, including on-premises data
>sources or other cloud systems using SQL or Python. `Athena` is built on open-source `Trino`
>and `Presto` engines and `Apache Spark` frameworks, with no provisioning or configuration effort required.

This notebook goes over how to load documents from `AWS Athena`.

## Setting up

Follow [instructions to set up an AWS account](https://docs.aws.amazon.com/athena/latest/ug/setting-up.html).

Install a python library:
"""
logger.info("# Athena")

# ! pip install boto3

"""
## Example
"""
logger.info("## Example")


database_name = "my_database"
s3_output_path = "s3://my_bucket/query_results/"
query = "SELECT * FROM my_table"
profile_name = "my_profile"

loader = AthenaLoader(
    query=query,
    database=database_name,
    s3_output_uri=s3_output_path,
    profile_name=profile_name,
)

documents = loader.load()
logger.debug(documents)

"""
Example with metadata columns
"""
logger.info("Example with metadata columns")

database_name = "my_database"
s3_output_path = "s3://my_bucket/query_results/"
query = "SELECT * FROM my_table"
profile_name = "my_profile"
metadata_columns = ["_row", "_created_at"]

loader = AthenaLoader(
    query=query,
    database=database_name,
    s3_output_uri=s3_output_path,
    profile_name=profile_name,
    metadata_columns=metadata_columns,
)

documents = loader.load()
logger.debug(documents)

logger.info("\n\n[DONE]", bright=True)