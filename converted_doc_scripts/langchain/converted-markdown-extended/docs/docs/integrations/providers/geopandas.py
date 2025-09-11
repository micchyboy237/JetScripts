from jet.logger import logger
from langchain_community.document_loaders import OpenCityDataLoader
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
# Geopandas

>[GeoPandas](https://geopandas.org/) is an open source project to make working
> with geospatial data in python easier. `GeoPandas` extends the datatypes used by
> `pandas` to allow spatial operations on geometric types.
> Geometric operations are performed by `shapely`.


## Installation and Setup

We have to install several python packages.
"""
logger.info("# Geopandas")

pip install -U sodapy pandas geopandas

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/geopandas).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)