from jet.logger import logger
from langchain_community.document_loaders import GeoDataFrameLoader
from langchain_community.document_loaders import OpenCityDataLoader
import ast
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
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

[Geopandas](https://geopandas.org/en/stable/index.html) is an open-source project to make working with geospatial data in python easier. 

GeoPandas extends the datatypes used by pandas to allow spatial operations on geometric types. 

Geometric operations are performed by shapely. Geopandas further depends on fiona for file access and matplotlib for plotting.

LLM applications (chat, QA) that utilize geospatial data are an interesting area for exploration.
"""
logger.info("# Geopandas")

# %pip install --upgrade --quiet  sodapy
# %pip install --upgrade --quiet  pandas
# %pip install --upgrade --quiet  geopandas



"""
Create a GeoPandas dataframe from [`Open City Data`](/docs/integrations/document_loaders/open_city_data) as an example input.
"""
logger.info("Create a GeoPandas dataframe from [`Open City Data`](/docs/integrations/document_loaders/open_city_data) as an example input.")

dataset = "tmnf-yvry"  # San Francisco crime data
loader = OpenCityDataLoader(city_id="data.sfgov.org", dataset_id=dataset, limit=5000)
docs = loader.load()

df = pd.DataFrame([ast.literal_eval(d.page_content) for d in docs])

df["Latitude"] = df["location"].apply(lambda loc: loc["coordinates"][1])
df["Longitude"] = df["location"].apply(lambda loc: loc["coordinates"][0])

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)

gdf = gdf[
    (gdf["Longitude"] >= -123.173825)
    & (gdf["Longitude"] <= -122.281780)
    & (gdf["Latitude"] >= 37.623983)
    & (gdf["Latitude"] <= 37.929824)
]

"""
Visualization of the sample of SF crime data.
"""
logger.info("Visualization of the sample of SF crime data.")


sf = gpd.read_file("https://data.sfgov.org/resource/3psu-pn9h.geojson")

fig, ax = plt.subplots(figsize=(10, 10))
sf.plot(ax=ax, color="white", edgecolor="black")
gdf.plot(ax=ax, color="red", markersize=5)
plt.show()

"""
Load GeoPandas dataframe as a `Document` for downstream processing (embedding, chat, etc). 

The `geometry` will be the default `page_content` columns, and all other columns are placed in `metadata`.

But, we can specify the `page_content_column`.
"""
logger.info("Load GeoPandas dataframe as a `Document` for downstream processing (embedding, chat, etc).")


loader = GeoDataFrameLoader(data_frame=gdf, page_content_column="geometry")
docs = loader.load()

docs[0]

logger.info("\n\n[DONE]", bright=True)