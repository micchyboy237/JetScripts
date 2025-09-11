from jet.logger import logger
from langchain_text_splitters import RecursiveJsonSplitter
import json
import os
import requests
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
# How to split JSON data

This json splitter [splits](/docs/concepts/text_splitters/) json data while allowing control over chunk sizes. It traverses json data depth first and builds smaller json chunks. It attempts to keep nested json objects whole but will split them if needed to keep chunks between a min_chunk_size and the max_chunk_size.

If the value is not a nested json, but rather a very large string the string will not be split. If you need a hard cap on the chunk size consider composing this with a Recursive Text splitter on those chunks. There is an optional pre-processing step to split lists, by first converting them to json (dict) and then splitting them as such.

1. How the text is split: json value.
2. How the chunk size is measured: by number of characters.
"""
logger.info("# How to split JSON data")

# %pip install -qU langchain-text-splitters

"""
First we load some json data:
"""
logger.info("First we load some json data:")



json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

"""
## Basic usage

Specify `max_chunk_size` to constrain chunk sizes:
"""
logger.info("## Basic usage")


splitter = RecursiveJsonSplitter(max_chunk_size=300)

"""
To obtain json chunks, use the `.split_json` method:
"""
logger.info("To obtain json chunks, use the `.split_json` method:")

json_chunks = splitter.split_json(json_data=json_data)

for chunk in json_chunks[:3]:
    logger.debug(chunk)

"""
To obtain LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects, use the `.create_documents` method:
"""
logger.info("To obtain LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects, use the `.create_documents` method:")

docs = splitter.create_documents(texts=[json_data])

for doc in docs[:3]:
    logger.debug(doc)

"""
Or use `.split_text` to obtain string content directly:
"""
logger.info("Or use `.split_text` to obtain string content directly:")

texts = splitter.split_text(json_data=json_data)

logger.debug(texts[0])
logger.debug(texts[1])

"""
## How to manage chunk sizes from list content

Note that one of the chunks in this example is larger than the specified `max_chunk_size` of 300. Reviewing one of these chunks that was bigger we see there is a list object there:
"""
logger.info("## How to manage chunk sizes from list content")

logger.debug([len(text) for text in texts][:10])
logger.debug()
logger.debug(texts[3])

"""
The json splitter by default does not split lists.

Specify `convert_lists=True` to preprocess the json, converting list content to dicts with `index:item` as `key:val` pairs:
"""
logger.info("The json splitter by default does not split lists.")

texts = splitter.split_text(json_data=json_data, convert_lists=True)

"""
Let's look at the size of the chunks. Now they are all under the max
"""
logger.info("Let's look at the size of the chunks. Now they are all under the max")

logger.debug([len(text) for text in texts][:10])

"""
The list has been converted to a dict, but retains all the needed contextual information even if split into many chunks:
"""
logger.info("The list has been converted to a dict, but retains all the needed contextual information even if split into many chunks:")

logger.debug(texts[1])

docs[1]

logger.info("\n\n[DONE]", bright=True)