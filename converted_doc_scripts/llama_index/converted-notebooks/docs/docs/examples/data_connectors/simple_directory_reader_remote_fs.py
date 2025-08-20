import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from s3fs import S3FileSystem
import boto3
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/simple_directory_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Simple Directory Reader over a Remote FileSystem

The `SimpleDirectoryReader` is the most commonly used data connector that _just works_.  
By default, it can be used to parse a variety of file-types on your local filesystem into a list of `Document` objects.
Additionaly, it can also be configured to read from a remote filesystem just as easily! This is made possible through the [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/index.html) protocol.

This notebook will take you through an example of using `SimpleDirectoryReader` to load documents from an S3 bucket. You can either run this against an actual S3 bucket, or a locally emulated S3 bucket via [LocalStack](https://www.localstack.cloud/).

### Get Started

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Simple Directory Reader over a Remote FileSystem")

# !pip install llama-index s3fs boto3

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay1.txt'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay2.txt'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay3.txt'


endpoint_url = (
    "http://localhost:4566"  # use this line if you are using S3 via localstack
)
bucket_name = "llama-index-test-bucket"
s3 = boto3.resource("s3", endpoint_url=endpoint_url)
s3.create_bucket(Bucket=bucket_name)
bucket = s3.Bucket(bucket_name)
bucket.upload_file(
    f"{GENERATED_DIR}/paul_graham/paul_graham_essay1.txt", "essays/paul_graham_essay1.txt"
)
bucket.upload_file(
    f"{GENERATED_DIR}/paul_graham/paul_graham_essay2.txt",
    "essays/more_essays/paul_graham_essay2.txt",
)
bucket.upload_file(
    f"{GENERATED_DIR}/paul_graham/paul_graham_essay3.txt",
    "essays/even_more_essays/paul_graham_essay3.txt",
)



s3_fs = S3FileSystem(anon=False, endpoint_url=endpoint_url)

"""
Load specific files
"""
logger.info("Load specific files")

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
    recursive=True,  # recursively searches all subdirectories
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

for idx, doc in enumerate(docs):
    logger.debug(f"{idx} - {doc.metadata}")

docs = reader.load_data()
logger.debug(f"Loaded {len(docs)} docs")

for idx, doc in enumerate(docs):
    logger.debug(f"{idx} - {doc.metadata}")

"""
Create an iterator to load files and process them as they load
"""
logger.info("Create an iterator to load files and process them as they load")

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
    recursive=True,
)

all_docs = []
for docs in reader.iter_data():
    for doc in docs:
        doc.text = doc.text.upper()
        all_docs.append(doc)

logger.debug(len(all_docs))

"""
Exclude specific patterns on the remote FS
"""
logger.info("Exclude specific patterns on the remote FS")

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
    recursive=True,
    exclude=["essays/more_essays/*"],
)

all_docs = []
for docs in reader.iter_data():
    for doc in docs:
        doc.text = doc.text.upper()
        all_docs.append(doc)

logger.debug(len(all_docs))
all_docs

"""
Async execution is available through `aload_data`
"""
logger.info("Async execution is available through `aload_data`")

# import nest_asyncio

# nest_asyncio.apply()

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
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

logger.info("\n\n[DONE]", bright=True)