from jet.logger import logger
from langchain_community.embeddings import XinferenceEmbeddings
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
# Xorbits inference (Xinference)

This notebook goes over how to use Xinference embeddings within LangChain

## Installation

Install `Xinference` through PyPI:
"""
logger.info("# Xorbits inference (Xinference)")

# %pip install --upgrade --quiet  "xinference[all]"

"""
## Deploy Xinference Locally or in a Distributed Cluster.

For local deployment, run `xinference`. 

To deploy Xinference in a cluster, first start an Xinference supervisor using the `xinference-supervisor`. You can also use the option -p to specify the port and -H to specify the host. The default port is 9997.

Then, start the Xinference workers using `xinference-worker` on each server you want to run them on. 

You can consult the README file from [Xinference](https://github.com/xorbitsai/inference) for more information.

## Wrapper

To use Xinference with LangChain, you need to first launch a model. You can use command line interface (CLI) to do so:
"""
logger.info("## Deploy Xinference Locally or in a Distributed Cluster.")

# !xinference launch -n vicuna-v1.3 -f ggmlv3 -q q4_0

"""
A model UID is returned for you to use. Now you can use Xinference embeddings with LangChain:
"""
logger.info("A model UID is returned for you to use. Now you can use Xinference embeddings with LangChain:")


xinference = XinferenceEmbeddings(
    server_url="http://0.0.0.0:9997", model_uid="915845ee-2a04-11ee-8ed4-d29396a3f064"
)

query_result = xinference.embed_query("This is a test query")

doc_result = xinference.embed_documents(["text A", "text B"])

"""
Lastly, terminate the model when you do not need to use it:
"""
logger.info("Lastly, terminate the model when you do not need to use it:")

# !xinference terminate --model-uid "915845ee-2a04-11ee-8ed4-d29396a3f064"

logger.info("\n\n[DONE]", bright=True)