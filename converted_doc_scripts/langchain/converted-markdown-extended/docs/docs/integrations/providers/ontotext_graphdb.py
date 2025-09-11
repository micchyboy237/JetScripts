from jet.logger import logger
from langchain.chains import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph
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
# Ontotext GraphDB

>[Ontotext GraphDB](https://graphdb.ontotext.com/) is a graph database and knowledge discovery tool compliant with RDF and SPARQL.

## Dependencies

Install the [rdflib](https://github.com/RDFLib/rdflib) package with
"""
logger.info("# Ontotext GraphDB")

pip install rdflib==7.0.0

"""
## Graph QA Chain

Connect your GraphDB Database with a chat model to get insights on your data.

See the notebook example [here](/docs/integrations/graphs/ontotext).
"""
logger.info("## Graph QA Chain")


logger.info("\n\n[DONE]", bright=True)