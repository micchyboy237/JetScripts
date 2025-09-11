from jet.logger import logger
from langchain_community.tools import GooglePlacesTool
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
# Google Places

This notebook goes through how to use Google Places API
"""
logger.info("# Google Places")

# %pip install --upgrade --quiet  googlemaps langchain-community


os.environ["GPLACES_API_KEY"] = ""


places = GooglePlacesTool()

places.run("al fornos")

logger.info("\n\n[DONE]", bright=True)