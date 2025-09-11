from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.retrievers import (
PineconeHybridSearchRetriever,
)
from pinecone import Pinecone, ServerlessSpec
from pinecone_notebooks.colab import Authenticate
from pinecone_text.sparse import BM25Encoder
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
# Pinecone Hybrid Search

>[Pinecone](https://docs.pinecone.io/docs/overview) is a vector database with broad functionality.

This notebook goes over how to use a retriever that under the hood uses Pinecone and Hybrid Search.

The logic of this retriever is taken from [this documentation](https://docs.pinecone.io/docs/hybrid-search)

To use Pinecone, you must have an API key and an Environment. 
Here are the [installation instructions](https://docs.pinecone.io/docs/quickstart).
"""
logger.info("# Pinecone Hybrid Search")

# %pip install --upgrade --quiet  pinecone pinecone-text pinecone-notebooks


Authenticate()


api_key = os.environ["PINECONE_API_KEY"]


"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info("We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
## Setup Pinecone

You should only have to do this part once.
"""
logger.info("## Setup Pinecone")



index_name = "langchain-pinecone-hybrid-search"

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

"""
Now that the index is created, we can use it.
"""
logger.info("Now that the index is created, we can use it.")

index = pc.Index(index_name)

"""
## Get embeddings and sparse encoders

Embeddings are used for the dense vectors, tokenizer is used for the sparse vector
"""
logger.info("## Get embeddings and sparse encoders")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
To encode the text to sparse values you can either choose SPLADE or BM25. For out of domain tasks we recommend using BM25.

For more information about the sparse encoders you can checkout pinecone-text library [docs](https://pinecone-io.github.io/pinecone-text/pinecone_text.html).
"""
logger.info("To encode the text to sparse values you can either choose SPLADE or BM25. For out of domain tasks we recommend using BM25.")



bm25_encoder = BM25Encoder().default()

"""
The above code is using default tfids values. It's highly recommended to fit the tf-idf values to your own corpus. You can do it as follow:

```python
corpus = ["foo", "bar", "world", "hello"]

# fit tf-idf values on your corpus
bm25_encoder.fit(corpus)

# store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")
```

## Load Retriever

We can now construct the retriever!
"""
logger.info("# fit tf-idf values on your corpus")

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)

"""
## Add texts (if necessary)

We can optionally add texts to the retriever (if they aren't already in there)
"""
logger.info("## Add texts (if necessary)")

retriever.add_texts(["foo", "bar", "world", "hello"])

"""
## Use Retriever

We can now use the retriever!
"""
logger.info("## Use Retriever")

result = retriever.invoke("foo")

result[0]

logger.info("\n\n[DONE]", bright=True)