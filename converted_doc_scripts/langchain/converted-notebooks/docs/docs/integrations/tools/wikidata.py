from jet.logger import logger
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
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
# Wikidata

>[Wikidata](https://wikidata.org/) is a free and open knowledge base that can be read and edited by both humans and machines. Wikidata is one of the world's largest open knowledge bases.

First, you need to install `wikibase-rest-api-client` and `mediawikiapi` python packages.
"""
logger.info("# Wikidata")

# %pip install --upgrade --quiet wikibase-rest-api-client mediawikiapi


wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

logger.debug(wikidata.run("Alan Turing"))

logger.info("\n\n[DONE]", bright=True)