from jet.logger import logger
from langchain_community.document_loaders.rspace import RSpaceLoader
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
This notebook shows how to use the RSpace document loader to import research notes and documents from RSpace Electronic
Lab Notebook into Langchain pipelines.

To start you'll need an RSpace account and an API key.

You can set up a free account at [https://community.researchspace.com](https://community.researchspace.com) or use your institutional RSpace.

You can get an RSpace API token from your account's profile page.
"""
logger.info("This notebook shows how to use the RSpace document loader to import research notes and documents from RSpace Electronic")

# %pip install --upgrade --quiet  rspace_client

"""
It's best to store your RSpace API key as an environment variable. 

    RSPACE_API_KEY=&lt;YOUR_KEY&gt;

You'll also need to set the URL of your RSpace installation e.g.

    RSPACE_URL=https://community.researchspace.com

If you use these exact environment variable names, they will be detected automatically.
"""
logger.info("It's best to store your RSpace API key as an environment variable.")


"""
You can import various items from RSpace:

* A single RSpace structured or basic document. This will map 1-1 to a Langchain document.
* A folder or noteook. All documents inside the notebook or folder are imported as Langchain documents. 
* If you have PDF files in the RSpace Gallery, these can be imported individually as well. Under the hood, Langchain's PDF loader will be used and this creates one Langchain document per PDF page.
"""
logger.info("You can import various items from RSpace:")

rspace_ids = ["NB1932027", "FL1921314", "SD1932029", "GL1932384"]
for rs_id in rspace_ids:
    loader = RSpaceLoader(global_id=rs_id)
    docs = loader.load()
    for doc in docs:
        logger.debug(doc.metadata)
        logger.debug(doc.page_content[:500])

"""
If you don't want to use the environment variables as above, you can pass these into the RSpaceLoader
"""
logger.info("If you don't want to use the environment variables as above, you can pass these into the RSpaceLoader")

loader = RSpaceLoader(
    global_id=rs_id, url="https://my.researchspace.com"
)

logger.info("\n\n[DONE]", bright=True)