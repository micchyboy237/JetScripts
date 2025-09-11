from jet.logger import logger
from langchain_core.embeddings import Embeddings
from langchain_parrot_link import ParrotLinkEmbeddings
from typing import List
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
# Custom Embeddings

LangChain is integrated with many [3rd party embedding models](/docs/integrations/text_embedding/). In this guide we'll show you how to create a custom Embedding class, in case a built-in one does not already exist. Embeddings are critical in natural language processing applications as they convert text into a numerical form that algorithms can understand, thereby enabling a wide range of applications such as similarity search, text classification, and clustering.

Implementing embeddings using the standard [Embeddings](https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.embeddings.Embeddings.html) interface will allow your embeddings to be utilized in existing `LangChain` abstractions (e.g., as the embeddings powering a [VectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html) or cached using [CacheBackedEmbeddings](/docs/how_to/caching_embeddings/)).

## Interface

The current `Embeddings` abstraction in LangChain is designed to operate on text data. In this implementation, the inputs are either single strings or lists of strings, and the outputs are lists of numerical arrays (vectors), where each vector represents
an embedding of the input text into some n-dimensional space.

Your custom embedding must implement the following methods:

| Method/Property                 | Description                                                                | Required/Optional |
|---------------------------------|----------------------------------------------------------------------------|-------------------|
| `embed_documents(texts)`        | Generates embeddings for a list of strings.                                | Required          |
| `embed_query(text)`             | Generates an embedding for a single text query.                            | Required          |
| `aembed_documents(texts)`       | Asynchronously generates embeddings for a list of strings.                 | Optional          |
| `aembed_query(text)`            | Asynchronously generates an embedding for a single text query.             | Optional          |

These methods ensure that your embedding model can be integrated seamlessly into the LangChain framework, providing both synchronous and asynchronous capabilities for scalability and performance optimization.


:::note
`Embeddings` do not currently implement the [Runnable](/docs/concepts/runnables/) interface and are also **not** instances of pydantic `BaseModel`.
:::

### Embedding queries vs documents

The `embed_query` and `embed_documents` methods are required. These methods both operate
on string inputs. The accessing of `Document.page_content` attributes is handled
by the vector store using the embedding model for legacy reasons.

`embed_query` takes in a single string and returns a single embedding as a list of floats.
If your model has different modes for embedding queries vs the underlying documents, you can
implement this method to handle that. 

`embed_documents` takes in a list of strings and returns a list of embeddings as a list of lists of floats.

:::note
`embed_documents` takes in a list of plain text, not a list of LangChain `Document` objects. The name of this method
may change in future versions of LangChain.
:::

## Implementation

As an example, we'll implement a simple embeddings model that returns a constant vector. This model is for illustrative purposes only.
"""
logger.info("# Custom Embeddings")




class ParrotLinkEmbeddings(Embeddings):
    """ParrotLink embedding model integration.

    Key init args — completion params:
        model: str
            Name of ParrotLink model to use.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python


            embed = ParrotLinkEmbeddings(
                model="...",
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python


    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python


    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)


        .. code-block:: python


    """

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [[0.5, 0.6, 0.7] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

"""
### Let's test it 🧪
"""
logger.info("### Let's test it 🧪")

embeddings = ParrotLinkEmbeddings("test-model")
logger.debug(embeddings.embed_documents(["Hello", "world"]))
logger.debug(embeddings.embed_query("Hello"))

"""
## Contributing

We welcome contributions of Embedding models to the LangChain code base.

If you aim to contribute an embedding model for a new provider (e.g., with a new set of dependencies or SDK), we encourage you to publish your implementation in a separate `langchain-*` integration package. This will enable you to appropriately manage dependencies and version your package. Please refer to our [contributing guide](/docs/contributing/how_to/integrations/) for a walkthrough of this process.
"""
logger.info("## Contributing")

logger.info("\n\n[DONE]", bright=True)