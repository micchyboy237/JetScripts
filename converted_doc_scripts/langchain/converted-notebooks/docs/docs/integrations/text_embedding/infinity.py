from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.embeddings import InfinityEmbeddings, InfinityEmbeddingsLocal
import numpy as np
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
# Infinity

`Infinity` allows to create `Embeddings` using a MIT-licensed Embedding Server. 

This notebook goes over how to use Langchain with Embeddings with the [Infinity Github Project](https://github.com/michaelfeil/infinity).

## Imports
"""
logger.info("# Infinity")


"""
# Option 1: Use infinity from Python

#### Optional: install infinity

To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
Install the torch and onnx dependencies. 

```bash
pip install infinity_emb[torch,optimum]
```
"""
logger.info("# Option 1: Use infinity from Python")

documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"

embeddings = InfinityEmbeddingsLocal(
    model="sentence-transformers/all-MiniLM-L6-v2",
    revision=None,
    batch_size=32,
    device="cuda",
)


async def embed():

    async with embeddings:
        
            documents_embedded = await embeddings.aembed_documents(documents)
            query_result = await embeddings.aembed_query(query)
            logger.debug("embeddings created successful")
    logger.success(format_json(result))
    return documents_embedded, query_result

documents_embedded, query_result = await embed()
logger.success(format_json(documents_embedded, query_result))


scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))

"""
# Option 2: Run the server, and connect via the API

#### Optional: Make sure to start the Infinity instance

To install infinity use the following command. For further details check out the [Docs on Github](https://github.com/michaelfeil/infinity).
```bash
pip install infinity_emb[all]
```

# Install the infinity package
%pip install --upgrade --quiet  infinity_emb[all]

Start up the server - best to be done from a separate terminal, not inside Jupyter Notebook

```bash
model=sentence-transformers/all-MiniLM-L6-v2
port=7797
infinity_emb --port $port --model-name-or-path $model
```

or alternativley just use docker:
```bash
model=sentence-transformers/all-MiniLM-L6-v2
port=7797
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
```

## Embed your documents using your Infinity instance
"""
logger.info("# Option 2: Run the server, and connect via the API")

documents = [
    "Baguette is a dish.",
    "Paris is the capital of France.",
    "numpy is a lib for linear algebra",
    "You escaped what I've escaped - You'd be in Paris getting fucked up too",
]
query = "Where is Paris?"

infinity_api_url = "http://localhost:7797/v1"
embeddings = InfinityEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2", infinity_api_url=infinity_api_url
)
try:
    documents_embedded = embeddings.embed_documents(documents)
    query_result = embeddings.embed_query(query)
    logger.debug("embeddings created successful")
except Exception as ex:
    logger.debug(
        "Make sure the infinity instance is running. Verify by clicking on "
        f"{infinity_api_url.replace('v1', 'docs')} Exception: {ex}. "
    )


scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))

logger.info("\n\n[DONE]", bright=True)