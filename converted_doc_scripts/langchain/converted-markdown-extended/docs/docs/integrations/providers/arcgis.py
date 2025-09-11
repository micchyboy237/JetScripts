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

>[ArcGIS](https://www.esri.com/en-us/arcgis/about-arcgis/overview) is a family of client,
> server and online geographic information system software developed and maintained by [Esri](https://www.esri.com/).
>
>`ArcGISLoader` uses the `arcgis` package.
> `arcgis` is a Python library for the vector and raster analysis, geocoding, map making,
> routing and directions. It administers, organizes and manages users,
> groups and information items in your GIS.
>It enables access to ready-to-use maps and curated geographic data from `Esri`
> and other authoritative sources, and works with your own data as well.

## Installation and Setup

We have to install the `arcgis` package.
"""
logger.info("# ArcGIS")

pip install -U arcgis

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/arcgis).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)