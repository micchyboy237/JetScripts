from jet.logger import logger
from langchain_community.document_loaders import WeatherDataLoader
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
# Weather

>[OpenWeatherMap](https://openweathermap.org/) is an open-source weather service provider

This loader fetches the weather data from the OpenWeatherMap's OneCall API, using the pyowm Python package. You must initialize the loader with your OpenWeatherMap API token and the names of the cities you want the weather data for.
"""
logger.info("# Weather")


# %pip install --upgrade --quiet  pyowm

# from getpass import getpass

# OPENWEATHERMAP_API_KEY = getpass()

loader = WeatherDataLoader.from_params(
    ["chennai", "vellore"], openweathermap_api_key=OPENWEATHERMAP_API_KEY
)

documents = loader.load()
documents

logger.info("\n\n[DONE]", bright=True)