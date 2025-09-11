from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.vectorstores import InfinispanVS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import csv
import gzip
import os
import shutil
import time


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
# Infinispan

Infinispan is an open-source key-value data grid, it can work as single node as well as distributed.

Vector search is supported since release 15.x
For more: [Infinispan Home](https://infinispan.org)
"""
logger.info("# Infinispan")

# %pip install sentence-transformers
# %pip install langchain
# %pip install langchain_core
# %pip install langchain_community

"""
# Setup

To run this demo we need a running Infinispan instance without authentication and a data file.
In the next three cells we're going to:
- download the data file
- create the configuration
- run Infinispan in docker
"""
logger.info("# Setup")

# %%bash
wget https://raw.githubusercontent.com/rigazilla/infinispan-vector/main/bbc_news.csv.gz

# %%bash
echo 'infinispan:
  cache-container:
    name: default
    transport:
      cluster: cluster
      stack: tcp
  server:
    interfaces:
      interface:
        name: public
        inet-address:
          value: 0.0.0.0
    socket-bindings:
      default-interface: public
      port-offset: 0
      socket-binding:
        name: default
        port: 11222
    endpoints:
      endpoint:
        socket-binding: default
        rest-connector:
' > infinispan-noauth.yaml

# !docker rm --force infinispanvs-demo
# !docker run -d --name infinispanvs-demo -v $(pwd):/user-config  -p 11222:11222 infinispan/server:15.0 -c /user-config/infinispan-noauth.yaml

"""
# The Code

## Pick up an embedding model

In this demo we're using
a HuggingFace embedding mode.
"""
logger.info("# The Code")


model_name = "sentence-transformers/all-MiniLM-L12-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)

"""
## Setup Infinispan cache

Infinispan is a very flexible key-value store, it can store raw bits as well as complex data type.
User has complete freedom in the datagrid configuration, but for simple data type everything is automatically
configured by the python layer. We take advantage of this feature so we can focus on our application.

## Prepare the data

In this demo we rely on the default configuration, thus texts, metadatas and vectors in the same cache, but other options are possible: i.e. content can be store somewhere else and vector store could contain only a reference to the actual content.
"""
logger.info("## Setup Infinispan cache")


with gzip.open("bbc_news.csv.gz", "rt", newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
    i = 0
    texts = []
    metas = []
    embeds = []
    for row in spamreader:
        text = row[0] + "." + row[4]
        texts.append(text)
        meta = {"text": row[4], "title": row[0]}
        metas.append(meta)
        i = i + 1
        if i >= 5000:
            break

"""
# Populate the vector store
"""
logger.info("# Populate the vector store")


ispnvs = InfinispanVS.from_texts(texts, hf, metas)

"""
# An helper func that prints the result documents

By default InfinispanVS returns the protobuf `ŧext` field in the `Document.page_content`
and all the remaining protobuf fields (except the vector) in the `metadata`. This behaviour is
configurable via lambda functions at setup.
"""
logger.info("# An helper func that prints the result documents")

def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        logger.debug("----" + str(i + 1) + "----")
        logger.debug("TITLE: " + res.metadata["title"])
        logger.debug(res.page_content)

"""
# Try it!!!

Below some sample queries
"""
logger.info("# Try it!!!")

docs = ispnvs.similarity_search("European nations", 5)
print_docs(docs)

print_docs(ispnvs.similarity_search("Milan fashion week begins", 2))

print_docs(ispnvs.similarity_search("Stock market is rising today", 4))

print_docs(ispnvs.similarity_search("Why cats are so viral?", 2))

print_docs(ispnvs.similarity_search("How to stay young", 5))

# !docker rm --force infinispanvs-demo

logger.info("\n\n[DONE]", bright=True)