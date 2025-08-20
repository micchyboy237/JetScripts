from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.legacy_office import LegacyOfficeReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/lagecy_office_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Legacy Office Reader

The `LegacyOfficeReader` is the reader for Word-97(.doc) files. Under the hood, it uses Apache Tika to parse the file.

### Get Started

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and the legacy office reader.

> Note: Apache Tika is a dependency of the legacy office reader and it requires Java to be installed and call-able via `java --version`.
> 
> For instance, on colab, you can install it with `!apt-get install default-jdk`. or on macOS, you can install it with `brew install openjdk`.
"""
logger.info("# Legacy Office Reader")

# %pip install llama-index-readers-legacy-office

"""
Prepare Data

So we need to prepare a .doc file for testing. Supposedly it's in `test_dir/harry_potter_lagacy.doc`
"""
logger.info("Prepare Data")


"""
**Option 1**: Load the file with `LegacyOfficeReader`
"""

file_path = "./test_dir/harry_potter_lagacy.doc"
reader = LegacyOfficeReader(
    excluded_embed_metadata_keys=["file_path", "file_name"],
    excluded_llm_metadata_keys=["file_type"],
)

docs = reader.load_data(file=file_path)
logger.debug(f"Loaded {len(docs)} docs")

"""
**Option 2**: Load the file with `SimpleDirectoryReader`

This is the path where we have `.doc` files together with other files in the same directory.

```python

reader = SimpleDirectoryReader(
    input_dir="./test_dir/",
    file_extractor={
        ".doc": LegacyOfficeReader(),
        }
)
```
"""
logger.info("This is the path where we have `.doc` files together with other files in the same directory.")

logger.info("\n\n[DONE]", bright=True)