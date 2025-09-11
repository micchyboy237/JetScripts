from jet.logger import logger
from langchain_community.document_loaders import MWDumpLoader
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
# MediaWikiDump

>[MediaWiki XML Dumps](https://www.mediawiki.org/wiki/Manual:Importing_XML_dumps) contain the content of a wiki
> (wiki pages with all their revisions), without the site-related data. A XML dump does not create a full backup
> of the wiki database, the dump does not contain user accounts, images, edit logs, etc.


## Installation and Setup

We need to install several python packages.

The `mediawiki-utilities` supports XML schema 0.11 in unmerged branches.
"""
logger.info("# MediaWikiDump")

pip install -qU git+https://github.com/mediawiki-utilities/python-mwtypes@updates_schema_0.11

"""
The `mediawiki-utilities mwxml` has a bug, fix PR pending.
"""
logger.info("The `mediawiki-utilities mwxml` has a bug, fix PR pending.")

pip install -qU git+https://github.com/gdedrouas/python-mwxml@xml_format_0.11
pip install -qU mwparserfromhell

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/mediawikidump).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)