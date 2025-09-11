from jet.logger import logger
from langchain_community.document_loaders import OutlookMessageLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
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
# Email

This notebook shows how to load email (`.eml`) or `Microsoft Outlook` (`.msg`) files.

Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

## Using Unstructured
"""
logger.info("# Email")

# %pip install --upgrade --quiet unstructured


loader = UnstructuredEmailLoader("./example_data/fake-email.eml")

data = loader.load()

data

"""
### Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.
"""
logger.info("### Retain Elements")

loader = UnstructuredEmailLoader("example_data/fake-email.eml", mode="elements")

data = loader.load()

data[0]

"""
### Processing Attachments

You can process attachments with `UnstructuredEmailLoader` by setting `process_attachments=True` in the constructor. By default, attachments will be partitioned using the `partition` function from `unstructured`. You can use a different partitioning function by passing the function to the `attachment_partitioner` kwarg.
"""
logger.info("### Processing Attachments")

loader = UnstructuredEmailLoader(
    "example_data/fake-email.eml",
    mode="elements",
    process_attachments=True,
)

data = loader.load()

data[0]

"""
## Using OutlookMessageLoader
"""
logger.info("## Using OutlookMessageLoader")

# %pip install --upgrade --quiet extract_msg


loader = OutlookMessageLoader("example_data/fake-email.msg")

data = loader.load()

data[0]

logger.info("\n\n[DONE]", bright=True)