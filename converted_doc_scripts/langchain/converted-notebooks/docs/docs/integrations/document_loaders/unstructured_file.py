from jet.logger import logger
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from unstructured_client import UnstructuredClient
from unstructured_client.utils import BackoffStrategy, RetryConfig
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
# Unstructured

This notebook covers how to use `Unstructured` [document loader](https://python.langchain.com/docs/concepts/document_loaders) to load files of many types. `Unstructured` currently supports loading of text files, powerpoints, html, pdfs, images, and more.

Please see [this guide](../../integrations/providers/unstructured.mdx) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
| :--- | :--- | :---: | :---: |  :---: |
| [UnstructuredLoader](https://python.langchain.com/api_reference/unstructured/document_loaders/langchain_unstructured.document_loaders.UnstructuredLoader.html) | [langchain-unstructured](https://python.langchain.com/api_reference/unstructured/index.html) | ✅ | ❌ | ✅ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| UnstructuredLoader | ✅ | ❌ | 

## Setup

### Credentials

By default, `langchain-unstructured` installs a smaller footprint that requires offloading of the partitioning logic to the Unstructured API, which requires an API key. If you use the local installation, you do not need an API key. To get your API key, head over to [this site](https://unstructured.io) and get an API key, and then set it in the cell below:
"""
logger.info("# Unstructured")

# import getpass

if "UNSTRUCTURED_API_KEY" not in os.environ:
#     os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass(
        "Enter your Unstructured API key: "
    )

"""
### Installation

#### Normal Installation

The following packages are required to run the rest of this notebook.
"""
logger.info("### Installation")

# %pip install --upgrade --quiet langchain-unstructured unstructured-client unstructured "unstructured[pdf]" python-magic

"""
#### Installation for Local

If you would like to run the partitioning logic locally, you will need to install a combination of system dependencies, as outlined in the [Unstructured documentation here](https://docs.unstructured.io/open-source/installation/full-installation).

For example, on Macs you can install the required dependencies with:

```bash
# base dependencies
brew install libmagic poppler tesseract

# If parsing xml / html documents:
brew install libxml2 libxslt
```

You can install the required `pip` dependencies needed for local with:

```bash
pip install "langchain-unstructured[local]"
```

## Initialization

The `UnstructuredLoader` allows loading from a variety of different file types. To read all about the `unstructured` package please refer to their [documentation](https://docs.unstructured.io/open-source/introduction/overview)/. In this example, we show loading from both a text file and a PDF file.
"""
logger.info("#### Installation for Local")


file_paths = [
    "./example_data/layout-parser-paper.pdf",
    "./example_data/state_of_the_union.txt",
]


loader = UnstructuredLoader(file_paths)

"""
## Load
"""
logger.info("## Load")

docs = loader.load()

docs[0]

logger.debug(docs[0].metadata)

"""
## Lazy Load
"""
logger.info("## Lazy Load")

pages = []
for doc in loader.lazy_load():
    pages.append(doc)

pages[0]

"""
## Post Processing

If you need to post process the `unstructured` elements after extraction, you can pass in a list of
`str` -> `str` functions to the `post_processors` kwarg when you instantiate the `UnstructuredLoader`. This applies to other Unstructured loaders as well. Below is an example.
"""
logger.info("## Post Processing")


loader = UnstructuredLoader(
    "./example_data/layout-parser-paper.pdf",
    post_processors=[clean_extra_whitespace],
)

docs = loader.load()

docs[5:10]

"""
## Unstructured API

If you want to get up and running with smaller packages and get the most up-to-date partitioning you can `pip install
unstructured-client` and `pip install langchain-unstructured`. For
more information about the `UnstructuredLoader`, refer to the
[Unstructured provider page](https://python.langchain.com/v0.1/docs/integrations/document_loaders/unstructured_file/).

The loader will process your document using the hosted Unstructured serverless API when you pass in
your `api_key` and set `partition_via_api=True`. You can generate a free
Unstructured API key [here](https://unstructured.io/api-key/).

Check out the instructions [here](https://github.com/Unstructured-IO/unstructured-api#dizzy-instructions-for-using-the-docker-image)
if you’d like to self-host the Unstructured API or run it locally.
"""
logger.info("## Unstructured API")


loader = UnstructuredLoader(
    file_path="example_data/fake.docx",
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    partition_via_api=True,
)

docs = loader.load()
docs[0]

"""
You can also batch multiple files through the Unstructured API in a single API using `UnstructuredLoader`.
"""
logger.info("You can also batch multiple files through the Unstructured API in a single API using `UnstructuredLoader`.")

loader = UnstructuredLoader(
    file_path=["example_data/fake.docx", "example_data/fake-email.eml"],
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    partition_via_api=True,
)

docs = loader.load()

logger.debug(docs[0].metadata["filename"], ": ", docs[0].page_content[:100])
logger.debug(docs[-1].metadata["filename"], ": ", docs[-1].page_content[:100])

"""
### Unstructured SDK Client

Partitioning with the Unstructured API relies on the [Unstructured SDK
Client](https://docs.unstructured.io/api-reference/api-services/accessing-unstructured-api).

If you want to customize the client, you will have to pass an `UnstructuredClient` instance to the `UnstructuredLoader`. Below is an example showing how you can customize features of the client such as using your own `requests.Session()`, passing an alternative `server_url`, and customizing the `RetryConfig` object. For more information about customizing the client or what additional parameters the sdk client accepts, refer to the [Unstructured Python SDK](https://docs.unstructured.io/api-reference/api-services/sdk-python) docs and the client section of the [API Parameters](https://docs.unstructured.io/api-reference/api-services/api-parameters) docs. Note that all API Parameters should be passed to the `UnstructuredLoader`.

<div class="alert alert-block alert-warning"><b>Warning:</b> The example below may not use the latest version of the UnstructuredClient and there could be breaking changes in future releases. For the latest examples, refer to the <a href="https://docs.unstructured.io/api-reference/api-services/sdk-python">Unstructured Python SDK</a> docs.</div>
"""
logger.info("### Unstructured SDK Client")


client = UnstructuredClient(
    api_key_auth=os.getenv(
        "UNSTRUCTURED_API_KEY"
    ),  # Note: the client API param is "api_key_auth" instead of "api_key"
    client=requests.Session(),  # Define your own requests session
    server_url="https://api.unstructuredapp.io/general/v0/general",  # Define your own api url
    retry_config=RetryConfig(
        strategy="backoff",
        retry_connection_errors=True,
        backoff=BackoffStrategy(
            initial_interval=500,
            max_interval=60000,
            exponent=1.5,
            max_elapsed_time=900000,
        ),
    ),  # Define your own retry config
)

loader = UnstructuredLoader(
    "./example_data/layout-parser-paper.pdf",
    partition_via_api=True,
    client=client,
    split_pdf_page=True,
    split_pdf_page_range=[1, 10],
)

docs = loader.load()

logger.debug(docs[0].metadata["filename"], ": ", docs[0].page_content[:100])

"""
## Chunking

The `UnstructuredLoader` does not support `mode` as parameter for grouping text like the older
loader `UnstructuredFileLoader` and others did. It instead supports "chunking". Chunking in
unstructured differs from other chunking mechanisms you may be familiar with that form chunks based
on plain-text features--character sequences like "\n\n" or "\n" that might indicate a paragraph
boundary or list-item boundary. Instead, all documents are split using specific knowledge about each
document format to partition the document into semantic units (document elements) and we only need to
resort to text-splitting when a single element exceeds the desired maximum chunk size. In general,
chunking combines consecutive elements to form chunks as large as possible without exceeding the
maximum chunk size. Chunking produces a sequence of CompositeElement, Table, or TableChunk elements.
Each “chunk” is an instance of one of these three types.

See this [page](https://docs.unstructured.io/open-source/core-functionality/chunking) for more
details about chunking options, but to reproduce the same behavior as `mode="single"`, you can set
`chunking_strategy="basic"`, `max_characters=<some-really-big-number>`, and `include_orig_elements=False`.
"""
logger.info("## Chunking")


loader = UnstructuredLoader(
    "./example_data/layout-parser-paper.pdf",
    chunking_strategy="basic",
    max_characters=1000000,
    include_orig_elements=False,
)

docs = loader.load()

logger.debug("Number of LangChain documents:", len(docs))
logger.debug("Length of text in the document:", len(docs[0].page_content))

"""
## Loading web pages

`UnstructuredLoader` accepts a `web_url` kwarg when run locally that populates the `url` parameter of the underlying Unstructured [partition](https://docs.unstructured.io/open-source/core-functionality/partitioning). This allows for the parsing of remotely hosted documents, such as HTML web pages.

Example usage:
"""
logger.info("## Loading web pages")


loader = UnstructuredLoader(web_url="https://www.example.com")
docs = loader.load()

for doc in docs:
    logger.debug(f"{doc}\n")

"""
## API reference

For detailed documentation of all `UnstructuredLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/unstructured/document_loaders/langchain_unstructured.document_loaders.UnstructuredLoader.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)