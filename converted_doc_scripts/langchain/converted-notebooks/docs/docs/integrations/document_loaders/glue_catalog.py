from jet.logger import logger
from langchain_community.document_loaders.glue_catalog import GlueCatalogLoader
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
# Glue Catalog


The [AWS Glue Data Catalog](https://docs.aws.amazon.com/en_en/glue/latest/dg/catalog-and-crawler.html) is a centralized metadata repository that allows you to manage, access, and share metadata about your data stored in AWS. It acts as a metadata store for your data assets, enabling various AWS services and your applications to query and connect to the data they need efficiently.

When you define data sources, transformations, and targets in AWS Glue, the metadata about these elements is stored in the Data Catalog. This includes information about data locations, schema definitions, runtime metrics, and more. It supports various data store types, such as Amazon S3, Amazon RDS, Amazon Redshift, and external databases compatible with JDBC. It is also directly integrated with Amazon Athena, Amazon Redshift Spectrum, and Amazon EMR, allowing these services to directly access and query the data.

The Langchain GlueCatalogLoader will get the schema of all tables inside the given Glue database in the same format as Pandas dtype.

## Setting up

- Follow [instructions to set up an AWS account](https://docs.aws.amazon.com/athena/latest/ug/setting-up.html).
- Install the boto3 library: `pip install boto3`

## Example
"""
logger.info("# Glue Catalog")


database_name = "my_database"
profile_name = "my_profile"

loader = GlueCatalogLoader(
    database=database_name,
    profile_name=profile_name,
)

schemas = loader.load()
logger.debug(schemas)

"""
## Example with table filtering

Table filtering allows you to selectively retrieve schema information for a specific subset of tables within a Glue database. Instead of loading the schemas for all tables, you can use the `table_filter` argument to specify exactly which tables you're interested in.
"""
logger.info("## Example with table filtering")


database_name = "my_database"
profile_name = "my_profile"
table_filter = ["table1", "table2", "table3"]

loader = GlueCatalogLoader(
    database=database_name, profile_name=profile_name, table_filter=table_filter
)

schemas = loader.load()
logger.debug(schemas)

logger.info("\n\n[DONE]", bright=True)