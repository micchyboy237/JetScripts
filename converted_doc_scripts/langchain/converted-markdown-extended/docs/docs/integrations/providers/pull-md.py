from jet.logger import logger
from langchain_pull_md import PullMdLoader
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
# PullMd Loader

>[PullMd](https://pull.md/) is a service that converts web pages into Markdown format. The `langchain-pull-md` package utilizes this service to convert URLs, especially those rendered with JavaScript frameworks like React, Angular, or Vue.js, into Markdown without the need for local rendering.

## Installation and Setup

To get started with `langchain-pull-md`, you need to install the package via pip:
"""
logger.info("# PullMd Loader")

pip install langchain-pull-md

"""
See the [usage example](/docs/integrations/document_loaders/pull_md) for detailed integration and usage instructions.

## Document Loader

The `PullMdLoader` class in `langchain-pull-md` provides an easy way to convert URLs to Markdown. It's particularly useful for loading content from modern web applications for use within LangChain's processing capabilities.
"""
logger.info("## Document Loader")


loader = PullMdLoader(url='https://example.com')

documents = loader.load()

for document in documents:
    logger.debug(document.page_content)

"""
This loader supports any URL and is particularly adept at handling sites built with dynamic JavaScript, making it a versatile tool for markdown extraction in data processing workflows.

## API Reference

For a comprehensive guide to all available functions and their parameters, visit the [API reference](https://github.com/chigwell/langchain-pull-md).

## Additional Resources

- [GitHub Repository](https://github.com/chigwell/langchain-pull-md)
- [PyPi Package](https://pypi.org/project/langchain-pull-md/)
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)