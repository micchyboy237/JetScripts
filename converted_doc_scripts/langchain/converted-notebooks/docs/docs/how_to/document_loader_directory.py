from jet.logger import logger
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain_community.document_loaders import TextLoader
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
# How to load documents from a directory

LangChain's [DirectoryLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html) implements functionality for reading files from disk into LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) objects. Here we demonstrate:

- How to load from a filesystem, including use of wildcard patterns;
- How to use multithreading for file I/O;
- How to use custom loader classes to parse specific file types (e.g., code);
- How to handle errors, such as those due to decoding.
"""
logger.info("# How to load documents from a directory")


"""
`DirectoryLoader` accepts a `loader_cls` kwarg, which defaults to [UnstructuredLoader](/docs/integrations/document_loaders/unstructured_file). [Unstructured](https://docs.unstructured.io/) supports parsing for a number of formats, such as PDF and HTML. Here we use it to read in a markdown (.md) file.

We can use the `glob` parameter to control which files to load. Note that here it doesn't load the `.rst` file or the `.html` files.
"""
logger.info("We can use the `glob` parameter to control which files to load. Note that here it doesn't load the `.rst` file or the `.html` files.")

loader = DirectoryLoader("../", glob="**/*.md")
docs = loader.load()
len(docs)

logger.debug(docs[0].page_content[:100])

"""
## Show a progress bar

By default a progress bar will not be shown. To show a progress bar, install the `tqdm` library (e.g. `pip install tqdm`), and set the `show_progress` parameter to `True`.
"""
logger.info("## Show a progress bar")

loader = DirectoryLoader("../", glob="**/*.md", show_progress=True)
docs = loader.load()

"""
## Use multithreading

By default the loading happens in one thread. In order to utilize several threads set the `use_multithreading` flag to true.
"""
logger.info("## Use multithreading")

loader = DirectoryLoader("../", glob="**/*.md", use_multithreading=True)
docs = loader.load()

"""
## Change loader class
By default this uses the `UnstructuredLoader` class. To customize the loader, specify the loader class in the `loader_cls` kwarg. Below we show an example using [TextLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html):
"""
logger.info("## Change loader class")


loader = DirectoryLoader("../", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

logger.debug(docs[0].page_content[:100])

"""
Notice that while the `UnstructuredLoader` parses Markdown headers, `TextLoader` does not.

If you need to load Python source code files, use the `PythonLoader`:
"""
logger.info("Notice that while the `UnstructuredLoader` parses Markdown headers, `TextLoader` does not.")


loader = DirectoryLoader("../../../../../", glob="**/*.py", loader_cls=PythonLoader)

"""
## Auto-detect file encodings with TextLoader

`DirectoryLoader` can help manage errors due to variations in file encodings. Below we will attempt to load in a collection of files, one of which includes non-UTF8 encodings.
"""
logger.info("## Auto-detect file encodings with TextLoader")

path = "../../../libs/langchain/tests/unit_tests/examples/"

loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)

"""
### A. Default Behavior

By default we raise an error:
"""
logger.info("### A. Default Behavior")

loader.load()

"""
The file `example-non-utf8.txt` uses a different encoding, so the `load()` function fails with a helpful message indicating which file failed decoding.

With the default behavior of `TextLoader` any failure to load any of the documents will fail the whole loading process and no documents are loaded.

### B. Silent fail

We can pass the parameter `silent_errors` to the `DirectoryLoader` to skip the files which could not be loaded and continue the load process.
"""
logger.info("### B. Silent fail")

loader = DirectoryLoader(
    path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True
)
docs = loader.load()

doc_sources = [doc.metadata["source"] for doc in docs]
doc_sources

"""
### C. Auto detect encodings

We can also ask `TextLoader` to auto detect the file encoding before failing, by passing the `autodetect_encoding` to the loader class.
"""
logger.info("### C. Auto detect encodings")

text_loader_kwargs = {"autodetect_encoding": True}
loader = DirectoryLoader(
    path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs
)
docs = loader.load()

doc_sources = [doc.metadata["source"] for doc in docs]
doc_sources

logger.info("\n\n[DONE]", bright=True)