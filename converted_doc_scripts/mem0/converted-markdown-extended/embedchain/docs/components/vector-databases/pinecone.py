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
title: Pinecone
---

## Overview

Install pinecone related dependencies using the following command:
"""
logger.info("## Overview")

pip install --upgrade 'pinecone-client pinecone-text'

"""
In order to use Pinecone as vector database, set the environment variable `PINECONE_API_KEY` which you can find on [Pinecone dashboard](https://app.pinecone.io/).

<CodeGroup>
"""
logger.info("In order to use Pinecone as vector database, set the environment variable `PINECONE_API_KEY` which you can find on [Pinecone dashboard](https://app.pinecone.io/).")


app = App.from_config(config_path="pod_config.yaml")
app = App.from_config(config_path="serverless_config.yaml")

"""

"""

vectordb:
  provider: pinecone
  config:
    metric: cosine
    vector_dimension: 1536
    index_name: my-pinecone-index
    pod_config:
      environment: gcp-starter
      metadata_config:
        indexed:
          - "url"
          - "hash"

"""

"""

vectordb:
  provider: pinecone
  config:
    metric: cosine
    vector_dimension: 1536
    index_name: my-pinecone-index
    serverless_config:
      cloud: aws
      region: us-west-2

"""
</CodeGroup>

<br />
<Note>
You can find more information about Pinecone configuration [here](https://docs.pinecone.io/docs/manage-indexes#create-a-pod-based-index).
You can also optionally provide `index_name` as a config param in yaml file to specify the index name. If not provided, the index name will be `{collection_name}-{vector_dimension}`.
</Note>

## Usage

### Hybrid search

Here is an example of how you can do hybrid search using Pinecone as a vector database through Embedchain.
"""
logger.info("## Usage")



config = {
    'app': {
        "config": {
            "id": "ec-docs-hybrid-search"
        }
    },
    'vectordb': {
        'provider': 'pinecone',
        'config': {
            'metric': 'dotproduct',
            'vector_dimension': 1536,
            'index_name': 'my-index',
            'serverless_config': {
                'cloud': 'aws',
                'region': 'us-west-2'
            },
            'hybrid_search': True, # Remember to set this for hybrid search
        }
    }
}

app = App.from_config(config=config)

app.add("/path/to/file.pdf", data_type="pdf_file", namespace="my-namespace")

app.query("<YOUR QUESTION HERE>", namespace="my-namespace")

app.chat("<YOUR QUESTION HERE>", namespace="my-namespace")

"""
Under the hood, Embedchain fetches the relevant chunks from the documents you added by doing hybrid search on the pinecone index.
If you have questions on how pinecone hybrid search works, please refer to their [offical documentation here](https://docs.pinecone.io/docs/hybrid-search).

<Snippet file="missing-vector-db-tip.mdx" />
"""
logger.info("Under the hood, Embedchain fetches the relevant chunks from the documents you added by doing hybrid search on the pinecone index.")

logger.info("\n\n[DONE]", bright=True)