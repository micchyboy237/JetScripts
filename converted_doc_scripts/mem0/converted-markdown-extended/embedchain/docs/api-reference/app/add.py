from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '📊 add'
---

`add()` method is used to load the data sources from different data sources to a RAG pipeline. You can find the signature below:

### Parameters

<ParamField path="source" type="str">
    The data to embed, can be a URL, local file or raw content, depending on the data type.. You can find the full list of supported data sources [here](/components/data-sources/overview).
</ParamField>
<ParamField path="data_type" type="str" optional>
    Type of data source. It can be automatically detected but user can force what data type to load as.
</ParamField>
<ParamField path="metadata" type="dict" optional>
    Any metadata that you want to store with the data source. Metadata is generally really useful for doing metadata filtering on top of semantic search to yield faster search and better results.
</ParamField>
<ParamField path="all_references" type="bool" optional>
    This parameter instructs Embedchain to retrieve all the context and information from the specified link, as well as from any reference links on the page.
</ParamField>

## Usage

### Load data from webpage
"""
logger.info("### Parameters")


app = App()
app.add("https://www.forbes.com/profile/elon-musk")

"""
### Load data from sitemap
"""
logger.info("### Load data from sitemap")


app = App()
app.add("https://python.langchain.com/sitemap.xml", data_type="sitemap")

"""
You can find complete list of supported data sources [here](/components/data-sources/overview).
"""
logger.info("You can find complete list of supported data sources [here](/components/data-sources/overview).")

logger.info("\n\n[DONE]", bright=True)