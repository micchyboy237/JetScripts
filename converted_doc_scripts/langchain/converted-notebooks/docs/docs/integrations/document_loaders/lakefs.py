from jet.logger import logger
from langchain_community.document_loaders import LakeFSLoader
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
# lakeFS

>[lakeFS](https://docs.lakefs.io/) provides scalable version control over the data lake, and uses Git-like semantics to create and access those versions.

This notebooks covers how to load document objects from a `lakeFS` path (whether it's an object or a prefix).

## Initializing the lakeFS loader

Replace `ENDPOINT`, `LAKEFS_ACCESS_KEY`, and `LAKEFS_SECRET_KEY` values with your own.
"""
logger.info("# lakeFS")


ENDPOINT = ""
LAKEFS_ACCESS_KEY = ""
LAKEFS_SECRET_KEY = ""

lakefs_loader = LakeFSLoader(
    lakefs_access_key=LAKEFS_ACCESS_KEY,
    lakefs_secret_key=LAKEFS_SECRET_KEY,
    lakefs_endpoint=ENDPOINT,
)

"""
## Specifying a path
You can specify a prefix or a complete object path to control which files to load.

Specify the repository, reference (branch, commit id, or tag), and path in the corresponding `REPO`, `REF`, and `PATH` to load the documents from:
"""
logger.info("## Specifying a path")

REPO = ""
REF = ""
PATH = ""

lakefs_loader.set_repo(REPO)
lakefs_loader.set_ref(REF)
lakefs_loader.set_path(PATH)

docs = lakefs_loader.load()
docs

logger.info("\n\n[DONE]", bright=True)