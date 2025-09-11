from jet.logger import logger
from langchain_community.embeddings import Model2vecEmbeddings
from model2vec import StaticModel
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
Model2Vec is a technique to turn any sentence transformer into a really small static model
[model2vec](https://github.com/MinishLab/model2vec) can be used to generate embeddings.

## Setup

```bash
pip install -U langchain-community
```

## Instantiation

Ensure that `model2vec` is installed

```bash
pip install -U model2vec
```

## Indexing and Retrieval
"""
logger.info("## Setup")


embeddings = Model2vecEmbeddings("minishlab/potion-base-8M")

query_text = "This is a test query."
query_result = embeddings.embed_query(query_text)

document_text = "This is a test document."
document_result = embeddings.embed_documents([document_text])

"""
## Direct Usage

Here's how you would directly make use of `model2vec`

```python

# Load a model from the HuggingFace hub (in this case the potion-base-8M model)
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Make embeddings
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])

# Make sequences of token embeddings
token_embeddings = model.encode_as_sequence(["It's dangerous to go alone!", "It's a secret to everybody."])
```

## API Reference

For more information check out the model2vec github [repo](https://github.com/MinishLab/model2vec)
"""
logger.info("## Direct Usage")

logger.info("\n\n[DONE]", bright=True)