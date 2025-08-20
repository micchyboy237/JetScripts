import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/simple_directory_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Simple Directory Reader

The `SimpleDirectoryReader` is the most commonly used data connector that _just works_.  
Simply pass in a input directory or a list of files.  
It will select the best file reader based on the file extensions.

### Get Started

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Simple Directory Reader")

# !pip install llama-index

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay1.txt'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay2.txt'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay3.txt'


"""
Load specific files
"""
logger.info("Load specific files")

reader = SimpleDirectoryReader(
    input_files=["/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay1.txt"]
)

docs = reader.load_data()
logger.debug(f"Loaded {len(docs)} docs")

"""
Load all (top-level) files from directory
"""
logger.info("Load all (top-level) files from directory")

reader = SimpleDirectoryReader(input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")

docs = reader.load_data()
logger.debug(f"Loaded {len(docs)} docs")

"""
Load all (recursive) files from directory
"""
logger.info("Load all (recursive) files from directory")

# !mkdir -p 'data/paul_graham/nested'
# !echo "This is a nested file" > 'data/paul_graham/nested/nested_file.md'

required_exts = [".md"]

reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=required_exts,
    recursive=True,
)

docs = reader.load_data()
logger.debug(f"Loaded {len(docs)} docs")

"""
Create an iterator to load files and process them as they load
"""
logger.info("Create an iterator to load files and process them as they load")

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
)

all_docs = []
for docs in reader.iter_data():
    for doc in docs:
        doc.text = doc.text.upper()
        all_docs.append(doc)

logger.debug(len(all_docs))

"""
Async execution is available through `aload_data`
"""
logger.info("Async execution is available through `aload_data`")

# import nest_asyncio

# nest_asyncio.apply()

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
)

async def run_async_code_bd5bb81d():
    async def run_async_code_ba509c5a():
        all_docs = await reader.aload_data()
        return all_docs
    all_docs = asyncio.run(run_async_code_ba509c5a())
    logger.success(format_json(all_docs))
    return all_docs
all_docs = asyncio.run(run_async_code_bd5bb81d())
logger.success(format_json(all_docs))

logger.debug(len(all_docs))

"""
## Full Configuration

This is the full list of arguments that can be passed to the `SimpleDirectoryReader`:

```python
class SimpleDirectoryReader(BaseReader):
    """
logger.info("## Full Configuration")Simple directory reader.

    Load files from file directory.
    Automatically select the best file reader given file extensions.

    Args:
        input_dir (str): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        exclude_empty (bool): Whether to exclude empty files (Optional).
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        filename_as_id (bool): Whether to use the filename as the document id.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
        to using the local file system. Can be changed to use any remote file system
        exposed via the fsspec interface.
    """
```
"""

logger.info("\n\n[DONE]", bright=True)