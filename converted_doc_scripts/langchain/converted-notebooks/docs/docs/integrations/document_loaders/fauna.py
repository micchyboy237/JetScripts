from jet.logger import logger
from langchain_community.document_loaders.fauna import FaunaLoader
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
# Fauna

>[Fauna](https://fauna.com/) is a Document Database.

Query `Fauna` documents
"""
logger.info("# Fauna")

# %pip install --upgrade --quiet  fauna

"""
## Query data example
"""
logger.info("## Query data example")


secret = "<enter-valid-fauna-secret>"
query = "Item.all()"  # Fauna query. Assumes that the collection is called "Item"
field = "text"  # The field that contains the page content. Assumes that the field is called "text"

loader = FaunaLoader(query, field, secret)
docs = loader.lazy_load()

for value in docs:
    logger.debug(value)

"""
### Query with Pagination
You get a `after` value if there are more data. You can get values after the curcor by passing in the `after` string in query. 

To learn more following [this link](https://fqlx-beta--fauna-docs.netlify.app/fqlx/beta/reference/schema_entities/set/static-paginate)
"""
logger.info("### Query with Pagination")

query = """
Item.paginate("hs+DzoPOg ... aY1hOohozrV7A")
Item.all()
"""
loader = FaunaLoader(query, field, secret)

logger.info("\n\n[DONE]", bright=True)