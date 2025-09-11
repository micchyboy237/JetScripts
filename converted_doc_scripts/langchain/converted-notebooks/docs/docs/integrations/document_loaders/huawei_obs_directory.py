from jet.logger import logger
from langchain_community.document_loaders import OBSDirectoryLoader
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
# Huawei OBS Directory
The following code demonstrates how to load objects from the Huawei OBS (Object Storage Service) as documents.
"""
logger.info("# Huawei OBS Directory")




endpoint = "your-endpoint"

config = {"ak": "your-access-key", "sk": "your-secret-key"}
loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint, config=config)

loader.load()

"""
## Specify a Prefix for Loading
If you want to load objects with a specific prefix from the bucket, you can use the following code:
"""
logger.info("## Specify a Prefix for Loading")

loader = OBSDirectoryLoader(
    "your-bucket-name", endpoint=endpoint, config=config, prefix="test_prefix"
)

loader.load()

"""
## Get Authentication Information from ECS
If your langchain is deployed on Huawei Cloud ECS and [Agency is set up](https://support.huaweicloud.com/intl/en-us/usermanual-ecs/ecs_03_0166.html#section7), the loader can directly get the security token from ECS without needing access key and secret key.
"""
logger.info("## Get Authentication Information from ECS")

config = {"get_token_from_ecs": True}
loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint, config=config)

loader.load()

"""
## Use a Public Bucket
If your bucket's bucket policy allows anonymous access (anonymous users have `listBucket` and `GetObject` permissions), you can directly load the objects without configuring the `config` parameter.
"""
logger.info("## Use a Public Bucket")

loader = OBSDirectoryLoader("your-bucket-name", endpoint=endpoint)

loader.load()

logger.info("\n\n[DONE]", bright=True)