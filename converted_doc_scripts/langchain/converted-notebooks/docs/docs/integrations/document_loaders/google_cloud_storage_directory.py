from jet.logger import logger
from langchain_google_community import GCSDirectoryLoader
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
# Google Cloud Storage Directory

>[Google Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.

This covers how to load document objects from an `Google Cloud Storage (GCS) directory (bucket)`.
"""
logger.info("# Google Cloud Storage Directory")

# %pip install --upgrade --quiet  langchain-google-community[gcs]


loader = GCSDirectoryLoader(project_name="aist", bucket="testing-hwc")

loader.load()

"""
## Specifying a prefix
You can also specify a prefix for more fine-grained control over what files to load -including loading all files from a specific folder-.
"""
logger.info("## Specifying a prefix")

loader = GCSDirectoryLoader(project_name="aist", bucket="testing-hwc", prefix="fake")

loader.load()

"""
## Continue on failure to load a single file
Files in a GCS bucket may cause errors during processing. Enable the `continue_on_failure=True` argument to allow silent failure. This means failure to process a single file will not break the function, it will log a warning instead.
"""
logger.info("## Continue on failure to load a single file")

loader = GCSDirectoryLoader(
    project_name="aist", bucket="testing-hwc", continue_on_failure=True
)

loader.load()

logger.info("\n\n[DONE]", bright=True)