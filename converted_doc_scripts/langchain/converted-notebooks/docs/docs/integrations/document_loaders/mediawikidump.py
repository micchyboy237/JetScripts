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
# MediaWiki Dump

>[MediaWiki XML Dumps](https://www.mediawiki.org/wiki/Manual:Importing_XML_dumps) contain the content of a wiki (wiki pages with all their revisions), without the site-related data. A XML dump does not create a full backup of the wiki database, the dump does not contain user accounts, images, edit logs, etc.

This covers how to load a MediaWiki XML dump file into a document format that we can use downstream.

It uses `mwxml` from `mediawiki-utilities` to dump and `mwparserfromhell` from `earwig` to parse MediaWiki wikicode.

Dump files can be obtained with dumpBackup.php or on the Special:Statistics page of the Wiki.
"""
logger.info("# MediaWiki Dump")

# %pip install --upgrade --quiet git+https://github.com/mediawiki-utilities/python-mwtypes@updates_schema_0.11
# %pip install --upgrade --quiet git+https://github.com/gdedrouas/python-mwxml@xml_format_0.11
# %pip install --upgrade --quiet mwparserfromhell


loader = MWDumpLoader(
    file_path="example_data/testmw_pages_current.xml",
    encoding="utf8",
    skip_redirects=True,  # will skip over pages that just redirect to other pages (or not if False)
    stop_on_error=False,  # will skip over pages that cause parsing errors (or not if False)
)
documents = loader.load()
logger.debug(f"You have {len(documents)} document(s) in your data ")

documents[:5]

logger.info("\n\n[DONE]", bright=True)