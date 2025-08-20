from dotenv import load_dotenv
from google.colab import userdata
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from pathlib import Path
from tempfile import mkdtemp
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/DoclingReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Docling Reader

## Overview

[Docling](https://github.com/DS4SD/docling) extracts PDF, DOCX, HTML, and other document formats into a rich representation (incl. layout, tables etc.), which it can export to Markdown or JSON.

Docling Reader and Docling Node Parser presented in this notebook seamlessly integrate Docling into LlamaIndex, enabling you to:
- use various document types in your LLM applications with ease and speed, and
- leverage Docling's rich format for advanced, document-native grounding.

## Setup

- 👉 For best conversion speed, use GPU acceleration whenever available; e.g. if running on Colab, use GPU-enabled runtime.
- Notebook uses HuggingFace's Inference API; for increased LLM quota, token can be provided via env var `HF_TOKEN`.
- Requirements can be installed as shown below (`--no-warn-conflicts` meant for Colab's pre-populated Python env; feel free to remove for stricter usage):
"""
logger.info("# Docling Reader")

# %pip install -q --progress-bar off --no-warn-conflicts llama-index-core llama-index-readers-docling llama-index-node-parser-docling llama-index-embeddings-huggingface llama-index-llms-huggingface-api llama-index-readers-file python-dotenv

"""
We can now define the main parameters:
"""
logger.info("We can now define the main parameters:")



def get_env_from_colab_or_os(key):
    try:

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)


load_dotenv()
EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
GEN_MODEL = HuggingFaceInferenceAPI(
    token=get_env_from_colab_or_os("HF_TOKEN"),
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
SOURCE = "https://arxiv.org/pdf/2408.09869"  # Docling Technical Report
QUERY = "Which are the main AI models in Docling?"

"""
## Using Markdown export

To create a simple RAG pipeline, we can:
- define a `DoclingReader`, which by default exports to Markdown, and
- use a standard node parser for these Markdown-based docs, e.g. a `MarkdownNodeParser`
"""
logger.info("## Using Markdown export")


reader = DoclingReader()
node_parser = MarkdownNodeParser()

index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
logger.debug(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])

"""
## Using Docling format

To leverage Docling's rich native format, we:
- create a `DoclingReader` with JSON export type, and
- employ a `DoclingNodeParser` in order to appropriately parse that Docling format.

Notice how the sources now also contain document-level grounding (e.g. page number or bounding box information):
"""
logger.info("## Using Docling format")


reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
node_parser = DoclingNodeParser()

index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
logger.debug(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])

"""
## With Simple Directory Reader

To demonstrate this usage pattern, we first set up a test document directory.
"""
logger.info("## With Simple Directory Reader")


tmp_dir_path = Path(mkdtemp())
r = requests.get(SOURCE)
with open(tmp_dir_path / f"{Path(SOURCE).name}.pdf", "wb") as out_file:
    out_file.write(r.content)

"""
Using the `reader` and `node_parser` definitions from any of the above variants, usage with `SimpleDirectoryReader` then looks as follows:
"""
logger.info("Using the `reader` and `node_parser` definitions from any of the above variants, usage with `SimpleDirectoryReader` then looks as follows:")


dir_reader = SimpleDirectoryReader(
    input_dir=tmp_dir_path,
    file_extractor={".pdf": reader},
)

index = VectorStoreIndex.from_documents(
    documents=dir_reader.load_data(SOURCE),
    transformations=[node_parser],
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
logger.debug(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
display([(n.text, n.metadata) for n in result.source_nodes])

logger.info("\n\n[DONE]", bright=True)