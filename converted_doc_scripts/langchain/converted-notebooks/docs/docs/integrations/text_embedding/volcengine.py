from jet.logger import logger
from langchain_community.embeddings import VolcanoEmbeddings
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
# Volc Engine

This notebook provides you with a guide on how to load the Volcano Embedding class.


## API Initialization

To use the LLM services based on [VolcEngine](https://www.volcengine.com/docs/82379/1099455), you have to initialize these parameters:

You could either choose to init the AK,SK in environment variables or init params:

```base
export VOLC_ACCESSKEY=XXX
export VOLC_SECRETKEY=XXX
```
"""
logger.info("# Volc Engine")

"""For basic init and call"""


os.environ["VOLC_ACCESSKEY"] = ""
os.environ["VOLC_SECRETKEY"] = ""

embed = VolcanoEmbeddings(volcano_ak="", volcano_sk="")
logger.debug("embed_documents result:")
res1 = embed.embed_documents(["foo", "bar"])
for r in res1:
    logger.debug("", r[:8])

logger.debug("embed_query result:")
res2 = embed.embed_query("foo")
logger.debug("", r[:8])

logger.info("\n\n[DONE]", bright=True)