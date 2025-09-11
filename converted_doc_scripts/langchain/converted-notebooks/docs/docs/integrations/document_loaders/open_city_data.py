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
# Open City Data

[Socrata](https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6) provides an API for city open data. 

For a dataset such as [SF crime](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry), see the `API` tab on top right. 

That provides you with the `dataset identifier`.

Use the dataset identifier to grab specific tables for a given city_id (`data.sfgov.org`) - 

E.g., `vw6y-z8j6` for [SF 311 data](https://dev.socrata.com/foundry/data.sfgov.org/vw6y-z8j6).

E.g., `tmnf-yvry` for [SF Police data](https://dev.socrata.com/foundry/data.sfgov.org/tmnf-yvry).
"""
logger.info("# Open City Data")

# %pip install --upgrade --quiet  sodapy


dataset = "vw6y-z8j6"  # 311 data
dataset = "tmnf-yvry"  # crime data
loader = OpenCityDataLoader(city_id="data.sfgov.org", dataset_id=dataset, limit=2000)

docs = loader.load()

eval(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)