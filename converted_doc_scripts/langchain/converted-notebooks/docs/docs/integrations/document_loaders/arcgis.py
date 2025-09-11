from jet.logger import logger
from langchain_community.document_loaders import ArcGISLoader
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
# ArcGIS

This notebook demonstrates the use of the `langchain_community.document_loaders.ArcGISLoader` class.

You will need to install the ArcGIS API for Python `arcgis` and, optionally, `bs4.BeautifulSoup`.

You can use an `arcgis.gis.GIS` object for authenticated data loading, or leave it blank to access public data.
"""
logger.info("# ArcGIS")


URL = "https://maps1.vcgov.org/arcgis/rest/services/Beaches/MapServer/7"
loader = ArcGISLoader(URL)

docs = loader.load()

"""
Let's measure loader latency.
"""
logger.info("Let's measure loader latency.")

# %%time

docs = loader.load()

docs[0].metadata

"""
### Retrieving Geometries  


If you want to retrieve feature geometries, you may do so with the `return_geometry` keyword.

Each document's geometry will be stored in its metadata dictionary.
"""
logger.info("### Retrieving Geometries")

loader_geom = ArcGISLoader(URL, return_geometry=True)

# %%time

docs = loader_geom.load()

docs[0].metadata["geometry"]

for doc in docs:
    logger.debug(doc.page_content)

logger.info("\n\n[DONE]", bright=True)