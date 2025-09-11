from git import Repo
from jet.logger import logger
from langchain_community.document_loaders import GitLoader
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
# Git

>[Git](https://en.wikipedia.org/wiki/Git) is a distributed version control system that tracks changes in any set of computer files, usually used for coordinating work among programmers collaboratively developing source code during software development.

This notebook shows how to load text files from `Git` repository.

## Load existing repository from disk
"""
logger.info("# Git")

# %pip install --upgrade --quiet  GitPython


repo = Repo.clone_from(
    "https://github.com/langchain-ai/langchain", to_path="./example_data/test_repo1"
)
branch = repo.head.reference


loader = GitLoader(repo_path="./example_data/test_repo1/", branch=branch)

data = loader.load()

len(data)

logger.debug(data[0])

"""
## Clone repository from url
"""
logger.info("## Clone repository from url")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./example_data/test_repo2/",
    branch="master",
)

data = loader.load()

len(data)

"""
## Filtering files to load
"""
logger.info("## Filtering files to load")


loader = GitLoader(
    repo_path="./example_data/test_repo1/",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

logger.info("\n\n[DONE]", bright=True)