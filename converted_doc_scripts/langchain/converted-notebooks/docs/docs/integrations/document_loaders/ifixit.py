from jet.logger import logger
from langchain_community.document_loaders import IFixitLoader
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
# iFixit

>[iFixit](https://www.ifixit.com) is the largest, open repair community on the web. The site contains nearly 100k repair manuals, 200k Questions & Answers on 42k devices, and all the data is licensed under CC-BY-NC-SA 3.0.

This loader will allow you to download the text of a repair guide, text of Q&A's and wikis from devices on `iFixit` using their open APIs.  It's incredibly useful for context related to technical documents and answers to questions about devices in the corpus of data on `iFixit`.
"""
logger.info("# iFixit")


loader = IFixitLoader("https://www.ifixit.com/Teardown/Banana+Teardown/811")
data = loader.load()

data

loader = IFixitLoader(
    "https://www.ifixit.com/Answers/View/318583/My+iPhone+6+is+typing+and+opening+apps+by+itself"
)
data = loader.load()

data

loader = IFixitLoader("https://www.ifixit.com/Device/Standard_iPad")
data = loader.load()

data

"""
## Searching iFixit using /suggest

If you're looking for a more general way to search iFixit based on a keyword or phrase, the /suggest endpoint will return content related to the search term, then the loader will load the content from each of the suggested items and prep and return the documents.
"""
logger.info("## Searching iFixit using /suggest")

data = IFixitLoader.load_suggestions("Banana")

data

logger.info("\n\n[DONE]", bright=True)