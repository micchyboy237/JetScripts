from jet.transformers.formatters import format_json
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import CodeBlock from "@theme/CodeBlock"
import EmbeddingTabs from "@theme/EmbeddingTabs"
import TabItem from '@theme/TabItem'
import Tabs from '@theme/Tabs'
import VectorStoreTabs from "@theme/VectorStoreTabs"
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
# Build a semantic search engine

This tutorial will familiarize you with LangChain's [document loader](/docs/concepts/document_loaders), [embedding](/docs/concepts/embedding_models), and [vector store](/docs/concepts/vectorstores) abstractions. These abstractions are designed to support retrieval of data--  from (vector) databases and other sources--  for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference, as in the case of retrieval-augmented generation, or [RAG](/docs/concepts/rag) (see our RAG tutorial [here](/docs/tutorials/rag)).

Here we will build a search engine over a PDF document. This will allow us to retrieve passages in the PDF that are similar to an input query.

## Concepts

This guide focuses on retrieval of text data. We will cover the following concepts:

- Documents and document loaders;
- Text splitters;
- Embeddings;
- Vector stores and retrievers.

## Setup

### Jupyter Notebook

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

This tutorial requires the `langchain-community` and `pypdf` packages:


<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install langchain-community pypdf</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain-community pypdf -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>


For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```


## Documents and Document Loaders

LangChain implements a [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) abstraction, which is intended to represent a unit of text and associated metadata. It has three attributes:

- `page_content`: a string representing the content;
- `metadata`: a dict containing arbitrary metadata;
- `id`: (optional) a string identifier for the document.

The `metadata` attribute can capture information about the source of the document, its relationship to other documents, and other information. Note that an individual `Document` object often represents a chunk of a larger document.

We can generate sample documents when desired:
```python

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

However, the LangChain ecosystem implements [document loaders](/docs/concepts/document_loaders) that [integrate with hundreds of common sources](/docs/integrations/document_loaders/). This makes it easy to incorporate data from these sources into your AI application.

### Loading documents

Let's load a PDF into a sequence of `Document` objects. There is a sample PDF in the LangChain repo [here](https://github.com/langchain-ai/langchain/tree/master/docs/docs/example_data) -- a 10-k filing for Nike from 2023. We can consult the LangChain documentation for [available PDF document loaders](/docs/integrations/document_loaders/#pdfs). Let's select [PyPDFLoader](/docs/integrations/document_loaders/pypdfloader/), which is fairly lightweight.
"""
logger.info("# Build a semantic search engine")


file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

logger.debug(len(docs))

"""
:::tip

See [this guide](/docs/how_to/document_loader_pdf/) for more detail on PDF document loaders.

:::

`PyPDFLoader` loads one `Document` object per PDF page. For each, we can easily access:

- The string content of the page;
- Metadata containing the file name and page number.
"""
logger.info(
    "See [this guide](/docs/how_to/document_loader_pdf/) for more detail on PDF document loaders.")

logger.debug(f"{docs[0].page_content[:200]}\n")
logger.debug(docs[0].metadata)

"""
### Splitting

For both information retrieval and downstream question-answering purposes, a page may be too coarse a representation. Our goal in the end will be to retrieve `Document` objects that answer an input query, and further splitting our PDF will help ensure that the meanings of relevant portions of the document are not "washed out" by surrounding text.

We can use [text splitters](/docs/concepts/text_splitters) for this purpose. Here we will use a simple text splitter that partitions based on characters. We will split our documents into chunks of 1000 characters
with 200 characters of overlap between chunks. The overlap helps
mitigate the possibility of separating a statement from important
context related to it. We use the
[RecursiveCharacterTextSplitter](/docs/how_to/recursive_text_splitter),
which will recursively split the document using common separators like
new lines until each chunk is the appropriate size. This is the
recommended text splitter for generic text use cases.

We set `add_start_index=True` so that the character index where each
split Document starts within the initial Document is preserved as
metadata attribute “start_index”.

See [this guide](/docs/how_to/document_loader_pdf/) for more detail about working with PDFs, including how to extract text from specific sections and images.
"""
logger.info("### Splitting")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)

"""
## Embeddings

Vector search is a common way to store and search over unstructured data (such as unstructured text). The idea is to store numeric vectors that are associated with the text. Given a query, we can [embed](/docs/concepts/embedding_models) it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to identify related text.

LangChain supports embeddings from [dozens of providers](/docs/integrations/text_embedding/). These models specify how text should be converted into a numeric vector. Let's select a model:


<EmbeddingTabs customVarName="embeddings" />
"""
logger.info("## Embeddings")


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
logger.debug(f"Generated vectors of length {len(vector_1)}\n")
logger.debug(vector_1[:10])

"""
Armed with a model for generating text embeddings, we can next store them in a special data structure that supports efficient similarity search.

## Vector stores

LangChain [VectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html) objects contain methods for adding text and `Document` objects to the store, and querying them using various similarity metrics. They are often initialized with [embedding](/docs/how_to/embed_text) models, which determine how text data is translated to numeric vectors.

LangChain includes a suite of [integrations](/docs/integrations/vectorstores) with different vector store technologies. Some vector stores are hosted by a provider (e.g., various cloud providers) and require specific credentials to use; some (such as [Postgres](/docs/integrations/vectorstores/pgvector)) run in separate infrastructure that can be run locally or via a third-party; others can run in-memory for lightweight workloads. Let's select a vector store:


<VectorStoreTabs/>
"""
logger.info("## Vector stores")


vector_store = Chroma(embedding_function=embeddings)

"""
Having instantiated our vector store, we can now index the documents.
"""
logger.info(
    "Having instantiated our vector store, we can now index the documents.")

ids = vector_store.add_documents(documents=all_splits)

"""
Note that most vector store implementations will allow you to connect to an existing vector store--  e.g., by providing a client, index name, or other information. See the documentation for a specific [integration](/docs/integrations/vectorstores) for more detail.

Once we've instantiated a `VectorStore` that contains documents, we can query it. [VectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html) includes methods for querying:
- Synchronously and asynchronously;
- By string query and by vector;
- With and without returning similarity scores;
- By similarity and [maximum marginal relevance](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.max_marginal_relevance_search) (to balance similarity with query to diversity in retrieved results).

The methods will generally include a list of [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) objects in their outputs.

### Usage

Embeddings typically represent text as a "dense" vector such that texts with similar meanings are geometrically close. This lets us retrieve relevant information just by passing in a question, without knowledge of any specific key-terms used in the document.

Return documents based on similarity to a string query:
"""
logger.info("### Usage")

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

logger.debug(results[0])

"""
Async query:
"""
logger.info("Async query:")

results = await vector_store.asimilarity_search("When was Nike incorporated?")
logger.success(format_json(results))

logger.debug(results[0])

"""
Return scores:
"""
logger.info("Return scores:")

results = vector_store.similarity_search_with_score(
    "What was Nike's revenue in 2023?")
doc, score = results[0]
logger.debug(f"Score: {score}\n")
logger.debug(doc)

"""
Return documents based on similarity to an embedded query:
"""
logger.info("Return documents based on similarity to an embedded query:")

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
logger.debug(results[0])

"""
Learn more:

- [API reference](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html)
- [How-to guide](/docs/how_to/vectorstores)
- [Integration-specific docs](/docs/integrations/vectorstores)

## Retrievers

LangChain `VectorStore` objects do not subclass [Runnable](https://python.langchain.com/api_reference/core/index.html#langchain-core-runnables). LangChain [Retrievers](https://python.langchain.com/api_reference/core/index.html#langchain-core-retrievers) are Runnables, so they implement a standard set of methods (e.g., synchronous and asynchronous `invoke` and `batch` operations). Although we can construct retrievers from vector stores, retrievers can interface with non-vector store sources of data, as well (such as external APIs).

We can create a simple version of this ourselves, without subclassing `Retriever`. If we choose what method we wish to use to retrieve documents, we can create a runnable easily. Below we will build one around the `similarity_search` method:
"""
logger.info("## Retrievers")


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

"""
Vectorstores implement an `as_retriever` method that will generate a Retriever, specifically a [VectorStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html). These retrievers include specific `search_type` and `search_kwargs` attributes that identify what methods of the underlying vector store to call, and how to parameterize them. For instance, we can replicate the above with the following:
"""
logger.info(
    "Vectorstores implement an `as_retriever` method that will generate a Retriever, specifically a [VectorStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html). These retrievers include specific `search_type` and `search_kwargs` attributes that identify what methods of the underlying vector store to call, and how to parameterize them. For instance, we can replicate the above with the following:")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

"""
`VectorStoreRetriever` supports search types of `"similarity"` (default), `"mmr"` (maximum marginal relevance, described above), and `"similarity_score_threshold"`. We can use the latter to threshold documents output by the retriever by similarity score.

Retrievers can easily be incorporated into more complex applications, such as [retrieval-augmented generation (RAG)](/docs/concepts/rag) applications that combine a given question with retrieved context into a prompt for a LLM. To learn more about building such an application, check out the [RAG tutorial](/docs/tutorials/rag) tutorial.

### Learn more:

Retrieval strategies can be rich and complex. For example:

- We can [infer hard rules and filters](/docs/how_to/self_query/) from a query (e.g., "using documents published after 2020");
- We can [return documents that are linked](/docs/how_to/parent_document_retriever/) to the retrieved context in some way (e.g., via some document taxonomy);
- We can generate [multiple embeddings](/docs/how_to/multi_vector) for each unit of context;
- We can [ensemble results](/docs/how_to/ensemble_retriever) from multiple retrievers;
- We can assign weights to documents, e.g., to weigh [recent documents](/docs/how_to/time_weighted_vectorstore/) higher.

The [retrievers](/docs/how_to#retrievers) section of the how-to guides covers these and other built-in retrieval strategies.

It is also straightforward to extend the [BaseRetriever](https://python.langchain.com/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html) class in order to implement custom retrievers. See our how-to guide [here](/docs/how_to/custom_retriever).


## Next steps

You've now seen how to build a semantic search engine over a PDF document.

For more on document loaders:

- [Conceptual guide](/docs/concepts/document_loaders)
- [How-to guides](/docs/how_to/#document-loaders)
- [Available integrations](/docs/integrations/document_loaders/)

For more on embeddings:

- [Conceptual guide](/docs/concepts/embedding_models/)
- [How-to guides](/docs/how_to/#embedding-models)
- [Available integrations](/docs/integrations/text_embedding/)

For more on vector stores:

- [Conceptual guide](/docs/concepts/vectorstores/)
- [How-to guides](/docs/how_to/#vector-stores)
- [Available integrations](/docs/integrations/vectorstores/)

For more on RAG, see:

- [Build a Retrieval Augmented Generation (RAG) App](/docs/tutorials/rag/)
- [Related how-to guides](/docs/how_to/#qa-with-rag)
"""
logger.info("### Learn more:")

logger.info("\n\n[DONE]", bright=True)
