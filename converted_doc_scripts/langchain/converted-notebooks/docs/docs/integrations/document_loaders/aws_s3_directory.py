from jet.logger import logger
from langchain_community.document_loaders import S3DirectoryLoader
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
# AWS S3 Directory

>[Amazon Simple Storage Service (Amazon S3)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html) is an object storage service

>[AWS S3 Directory](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html)

This covers how to load document objects from an `AWS S3 Directory` object.
"""
logger.info("# AWS S3 Directory")

# %pip install --upgrade --quiet  boto3


loader = S3DirectoryLoader("testing-hwc")

loader.load()

"""
## Specifying a prefix
You can also specify a prefix for more fine-grained control over what files to load.
"""
logger.info("## Specifying a prefix")

loader = S3DirectoryLoader("testing-hwc", prefix="fake")

loader.load()

"""
## Configuring the AWS Boto3 client
You can configure the AWS [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) client by passing
named arguments when creating the S3DirectoryLoader.
This is useful for instance when AWS credentials can't be set as environment variables.
See the [list of parameters](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session) that can be configured.
"""
logger.info("## Configuring the AWS Boto3 client")

loader = S3DirectoryLoader(
    "testing-hwc", aws_access_key_id="xxxx", aws_secret_access_key="yyyy"
)

loader.load()

logger.info("\n\n[DONE]", bright=True)