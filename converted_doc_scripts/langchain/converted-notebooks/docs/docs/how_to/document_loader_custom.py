from jet.transformers.formatters import format_json
from cloudpathlib import S3Client, S3Path
from jet.logger import logger
from langchain_community.document_loaders.blob_loaders import CloudBlobLoader
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Any
from typing import AsyncIterator, Iterator
import aiofiles
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
---
title: Custom Document Loader
sidebar_position: 10
---

# How to create a custom Document Loader

## Overview


Applications based on LLMs frequently entail extracting data from databases or files, like PDFs, and converting it into a format that LLMs can utilize. In LangChain, this usually involves creating Document objects, which encapsulate the extracted text (`page_content`) along with metadata—a dictionary containing details about the document, such as the author's name or the date of publication.

`Document` objects are often formatted into prompts that are fed into an LLM, allowing the LLM to use the information in the `Document` to generate a desired response (e.g., summarizing the document).
`Documents` can be either used immediately or indexed into a vectorstore for future retrieval and use.

The main abstractions for [Document Loading](/docs/concepts/document_loaders/) are:


| Component      | Description                    |
|----------------|--------------------------------|
| Document       | Contains `text` and `metadata` |
| BaseLoader     | Use to convert raw data into `Documents`  |
| Blob           | A representation of binary data that's located either in a file or in memory |
| BaseBlobParser | Logic to parse a `Blob` to yield `Document` objects |

This guide will demonstrate how to write custom document loading and file parsing logic; specifically, we'll see how to:

1. Create a standard document Loader by sub-classing from `BaseLoader`.
2. Create a parser using  `BaseBlobParser` and use it in conjunction with `Blob` and `BlobLoaders`. This is useful primarily when working with files.

## Standard Document Loader

A document loader can be implemented by sub-classing from a `BaseLoader` which provides a standard interface for loading documents.

### Interface 

| Method Name | Explanation |
|-------------|-------------|
| lazy_load   | Used to load documents one by one **lazily**. Use for production code. |
| alazy_load  | Async variant of `lazy_load` |
| load        | Used to load all the documents into memory **eagerly**. Use for prototyping or interactive work. |
| aload       | Used to load all the documents into memory **eagerly**. Use for prototyping or interactive work. **Added in 2024-04 to LangChain.** |

* The `load` methods is a convenience method meant solely for prototyping work -- it just invokes `list(self.lazy_load())`.
* The `alazy_load` has a default implementation that will delegate to `lazy_load`. If you're using async, we recommend overriding the default implementation and providing a native async implementation.

:::important
When implementing a document loader do **NOT** provide parameters via the `lazy_load` or `alazy_load` methods.

All configuration is expected to be passed through the initializer (__init__). This was a design choice made by LangChain to make sure that once a document loader has been instantiated it has all the information needed to load documents.
:::

### Installation

Install **langchain-core** and **langchain_community**.
"""
logger.info("# How to create a custom Document Loader")

# %pip install -qU langchain_core langchain_community

"""
### Implementation

Let's create an example of a standard document loader that loads a file and creates a document from each line in the file.
"""
logger.info("### Implementation")




class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads a file line by line."""

        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
                line_number = 0
                async for line in f:
                    yield Document(
                        page_content=line,
                        metadata={"line_number": line_number, "source": self.file_path},
                    )
                    line_number += 1
        logger.success(format_json(result))

"""
### Test 🧪


To test out the document loader, we need a file with some quality content.
"""
logger.info("### Test 🧪")

with open("./meow.txt", "w", encoding="utf-8") as f:
    quality_content = "meow meow🐱 \n meow meow🐱 \n meow😻😻"
    f.write(quality_content)

loader = CustomDocumentLoader("./meow.txt")

# %pip install -q aiofiles

for doc in loader.lazy_load():
    logger.debug()
    logger.debug(type(doc))
    logger.debug(doc)

async for doc in loader.alazy_load():
    logger.debug()
    logger.debug(type(doc))
    logger.debug(doc)

"""
:::tip

`load()` can be helpful in an interactive environment such as a jupyter notebook.

Avoid using it for production code since eager loading assumes that all the content
can fit into memory, which is not always the case, especially for enterprise data.
:::
"""
logger.info("Avoid using it for production code since eager loading assumes that all the content")

loader.load()

"""
## Working with Files

Many document loaders involve parsing files. The difference between such loaders usually stems from how the file is parsed, rather than how the file is loaded. For example, you can use `open` to read the binary content of either a PDF or a markdown file, but you need different parsing logic to convert that binary data into text.

As a result, it can be helpful to decouple the parsing logic from the loading logic, which makes it easier to re-use a given parser regardless of how the data was loaded.

### BaseBlobParser

A `BaseBlobParser` is an interface that accepts a `blob` and outputs a list of `Document` objects. A `blob` is a representation of data that lives either in memory or in a file. LangChain python has a `Blob` primitive which is inspired by the [Blob WebAPI spec](https://developer.mozilla.org/en-US/docs/Web/API/Blob).
"""
logger.info("## Working with Files")



class MyParser(BaseBlobParser):
    """A simple parser that creates a document from each line."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a blob into a document line by line."""
        line_number = 0
        with blob.as_bytes_io() as f:
            for line in f:
                line_number += 1
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": blob.source},
                )

blob = Blob.from_path("./meow.txt")
parser = MyParser()

list(parser.lazy_parse(blob))

"""
Using the **blob** API also allows one to load content directly from memory without having to read it from a file!
"""
logger.info("Using the **blob** API also allows one to load content directly from memory without having to read it from a file!")

blob = Blob(data=b"some data from memory\nmeow")
list(parser.lazy_parse(blob))

"""
### Blob

Let's take a quick look through some of the Blob API.
"""
logger.info("### Blob")

blob = Blob.from_path("./meow.txt", metadata={"foo": "bar"})

blob.encoding

blob.as_bytes()

blob.as_string()

blob.as_bytes_io()

blob.metadata

blob.source

"""
### Blob Loaders

While a parser encapsulates the logic needed to parse binary data into documents, *blob loaders* encapsulate the logic that's necessary to load blobs from a given storage location.

At the moment, `LangChain` supports `FileSystemBlobLoader` and `CloudBlobLoader`.

You can use the `FileSystemBlobLoader` to load blobs and then use the parser to parse them.
"""
logger.info("### Blob Loaders")


filesystem_blob_loader = FileSystemBlobLoader(
    path=".", glob="*.mdx", show_progress=True
)

# %pip install -q tqdm

parser = MyParser()
for blob in filesystem_blob_loader.yield_blobs():
    for doc in parser.lazy_parse(blob):
        logger.debug(doc)
        break

"""
Or, you can use `CloudBlobLoader` to load blobs from a cloud storage location (Supports s3://, az://, gs://, file:// schemes).
"""
logger.info("Or, you can use `CloudBlobLoader` to load blobs from a cloud storage location (Supports s3://, az://, gs://, file:// schemes).")

# %pip install -q 'cloudpathlib[s3]'

"""
```python

client = S3Client(no_sign_request=True)
client.set_as_default_client()

path = S3Path(
    "s3://bucket-01", client=client
)  # Supports s3://, az://, gs://, file:// schemes.

cloud_loader = CloudBlobLoader(path, glob="**/*.pdf", show_progress=True)

for blob in cloud_loader.yield_blobs():
    logger.debug(blob)
```

### Generic Loader

LangChain has a `GenericLoader` abstraction which composes a `BlobLoader` with a `BaseBlobParser`.

`GenericLoader` is meant to provide standardized classmethods that make it easy to use existing `BlobLoader` implementations. At the moment, the `FileSystemBlobLoader` and `CloudBlobLoader` are supported. See example below:
"""
logger.info("### Generic Loader")


generic_loader_filesystem = GenericLoader(
    blob_loader=filesystem_blob_loader, blob_parser=parser
)
for idx, doc in enumerate(generic_loader_filesystem.lazy_load()):
    if idx < 5:
        logger.debug(doc)

logger.debug("... output truncated for demo purposes")

"""
#### Custom Generic Loader

If you really like creating classes, you can sub-class and create a class to encapsulate the logic together.

You can sub-class from this class to load content using an existing loader.
"""
logger.info("#### Custom Generic Loader")



class MyCustomLoader(GenericLoader):
    @staticmethod
    def get_parser(**kwargs: Any) -> BaseBlobParser:
        """Override this method to associate a default parser with the class."""
        return MyParser()

loader = MyCustomLoader.from_filesystem(path=".", glob="*.mdx", show_progress=True)

for idx, doc in enumerate(loader.lazy_load()):
    if idx < 5:
        logger.debug(doc)

logger.debug("... output truncated for demo purposes")

logger.info("\n\n[DONE]", bright=True)