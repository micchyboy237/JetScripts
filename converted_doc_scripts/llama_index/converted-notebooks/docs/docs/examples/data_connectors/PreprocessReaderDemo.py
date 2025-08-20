from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.preprocess import PreprocessReader
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
# Preprocess
[Preprocess](https://preprocess.co) is an API service that splits any kind of document into optimal chunks of text for use in language model tasks.

Given documents in input `Preprocess` splits them into chunks of text that respect the layout and semantics of the original document.
We split the content by taking into account sections, paragraphs, lists, images, data tables, text tables, and slides, and following the content semantics for long texts.

Preprocess supports:
- PDFs
- Microsoft Office documents (Word, PowerPoint, Excel)
- OpenOffice documents (ods, odt, odp)
- HTML content (web pages, articles, emails)
- plain text.

`PreprocessLoader` interact the `Preprocess API library` to provide document conversion and chunking or to load already chunked files inside LangChain.

## Requirements
Install the `Python Preprocess library` if it is not already present:
"""
logger.info("# Preprocess")



"""
## Usage

To use Preprocess loader, you need to pass the `Preprocess API Key`. 
When initializing `PreprocessReader`, you should pass your `API Key`, if you don't have it yet, please ask for one at [support@preprocess.co](mailto:support@preprocess.co). Without an `API Key`, the loader will raise an error.

To chunk a file pass a valid filepath and the reader will start converting and chunking it.
`Preprocess` will chunk your files by applying an internal `Splitter`. For this reason, you should not parse the document into nodes using a `Splitter` or applying a `Splitter` while transforming documents in your `IngestionPipeline`.
"""
logger.info("## Usage")


loader = PreprocessReader(
    api_key="your-api-key", filepath="valid/path/to/file"
)

"""
If you want to handle the nodes directly:
"""
logger.info("If you want to handle the nodes directly:")

nodes = loader.get_nodes()

index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()

"""
By default `load_data()` returns a document for each chunk, remember to not apply any splitting to these documents
"""
logger.info("By default `load_data()` returns a document for each chunk, remember to not apply any splitting to these documents")

documents = loader.load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

data = loader.load()

"""
If you want to return only the extracted text and handle it with custom pipelines set `return_whole_document = True`
"""
logger.info("If you want to return only the extracted text and handle it with custom pipelines set `return_whole_document = True`")

document = loader.load_data(return_whole_document=True)

"""
If you want to load already chunked files you can do it via `process_id` passing it to the reader.
"""
logger.info("If you want to load already chunked files you can do it via `process_id` passing it to the reader.")

loader = PreprocessReader(api_key="your-api-key", process_id="your-process-id")

"""
## Other info

`PreprocessReader` is based on `pypreprocess` from [Preprocess](https://github.com/preprocess-co/pypreprocess) library.
For more information or other integration needs please check the [documentation](https://github.com/preprocess-co/pypreprocess).
"""
logger.info("## Other info")

logger.info("\n\n[DONE]", bright=True)