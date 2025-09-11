from jet.logger import logger
from langchain_community.document_loaders import AirtableLoader
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
# Airtable
"""
logger.info("# Airtable")

# %pip install --upgrade --quiet  pyairtable


"""
* Get your API key [here](https://support.airtable.com/docs/creating-and-using-api-keys-and-access-tokens).
* Get ID of your base [here](https://airtable.com/developers/web/api/introduction).
* Get your table ID from the table url as shown [here](https://www.highviewapps.com/kb/where-can-i-find-the-airtable-base-id-and-table-id/#:~:text=Both%20the%20Airtable%20Base%20ID,URL%20that%20begins%20with%20tbl).
"""


base_id = "xxx"
table_id = "xxx"
view = "xxx"  # optional

loader = AirtableLoader(api_key, table_id, base_id, view=view)
docs = loader.load()

"""
Returns each table row as `dict`.
"""
logger.info("Returns each table row as `dict`.")

len(docs)

eval(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)