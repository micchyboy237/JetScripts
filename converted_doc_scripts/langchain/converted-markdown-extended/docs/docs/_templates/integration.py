from jet.logger import logger
from langchain_community.chat_models import integration_class_REPLACE_ME
from langchain_community.document_loaders import integration_class_REPLACE_ME
from langchain_community.embeddings import integration_class_REPLACE_ME
from langchain_community.llms import integration_class_REPLACE_ME
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
[comment: Please, a reference example here "docs/integrations/arxiv.md"]::
[comment: Use this template to create a new .md file in "docs/integrations/"]::

# Title_REPLACE_ME

[comment: Only one Tile/H1 is allowed!]::

>
[comment: Description: After reading this description, a reader should decide if this integration is good enough to try/follow reading OR]::
[comment: go to read the next integration doc. ]::
[comment: Description should include a link to the source for follow reading.]::

## Installation and Setup

[comment: Installation and Setup: All necessary additional package installations and setups for Tokens, etc]::
"""
logger.info("# Title_REPLACE_ME")

pip install package_name_REPLACE_ME

"""
[comment: OR this text:]::

There isn't any special setup for it.

[comment: The next H2/## sections with names of the integration modules, like "LLM", "Text Embedding Models", etc]::
[comment: see "Modules" in the "index.html" page]::
[comment: Each H2 section should include a link to an example(s) and a Python code with the import of the integration class]::
[comment: Below are several example sections. Remove all unnecessary sections. Add all necessary sections not provided here.]::

## LLM

See a [usage example](/docs/integrations/llms/INCLUDE_REAL_NAME).
"""
logger.info("## LLM")


"""
## Text Embedding Models

See a [usage example](/docs/integrations/text_embedding/INCLUDE_REAL_NAME).
"""
logger.info("## Text Embedding Models")


"""
## Chat models

See a [usage example](/docs/integrations/chat/INCLUDE_REAL_NAME).
"""
logger.info("## Chat models")


"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/INCLUDE_REAL_NAME).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)