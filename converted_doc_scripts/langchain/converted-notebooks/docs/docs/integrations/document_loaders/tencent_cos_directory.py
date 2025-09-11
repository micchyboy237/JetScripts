from jet.logger import logger
from langchain_community.document_loaders import TencentCOSDirectoryLoader
from qcloud_cos import CosConfig
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
# Tencent COS Directory

>[Tencent Cloud Object Storage (COS)](https://www.tencentcloud.com/products/cos) is a distributed 
> storage service that enables you to store any amount of data from anywhere via HTTP/HTTPS protocols. 
> `COS` has no restrictions on data structure or format. It also has no bucket size limit and 
> partition management, making it suitable for virtually any use case, such as data delivery, 
> data processing, and data lakes. `COS` provides a web-based console, multi-language SDKs and APIs, 
> command line tool, and graphical tools. It works well with Amazon S3 APIs, allowing you to quickly 
> access community tools and plugins.


This covers how to load document objects from a `Tencent COS Directory`.
"""
logger.info("# Tencent COS Directory")

# %pip install --upgrade --quiet  cos-python-sdk-v5


conf = CosConfig(
    Region="your cos region",
    SecretId="your cos secret_id",
    SecretKey="your cos secret_key",
)
loader = TencentCOSDirectoryLoader(conf=conf, bucket="you_cos_bucket")

loader.load()

"""
## Specifying a prefix
You can also specify a prefix for more fine-grained control over what files to load.
"""
logger.info("## Specifying a prefix")

loader = TencentCOSDirectoryLoader(conf=conf, bucket="you_cos_bucket", prefix="fake")

loader.load()

logger.info("\n\n[DONE]", bright=True)