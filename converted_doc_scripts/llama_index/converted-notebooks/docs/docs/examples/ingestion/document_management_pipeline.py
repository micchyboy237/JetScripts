from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.docstore.redis import RedisDocumentStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Ingestion Pipeline + Document Management

Attaching a `docstore` to the ingestion pipeline will enable document management.

Using the `document.doc_id` or `node.ref_doc_id` as a grounding point, the ingestion pipeline will actively look for duplicate documents.

It works by
- Storing a map of `doc_id` -> `document_hash`
- If a duplicate `doc_id` is detected, and the hash has changed, the document will be re-processed
- If the hash has not changed, the document will be skipped in the pipeline

If we do not attach a vector store, we can only check for and remove duplicate inputs.

If a vector store is attached, we can also handle upserts! We have [another guide](/en/stable/examples/ingestion/redis_ingestion_pipeline) for upserts and vector stores.

## Create Seed Data
"""
logger.info("# Ingestion Pipeline + Document Management")

# %pip install llama-index-storage-docstore-redis
# %pip install llama-index-storage-docstore-mongodb
# %pip install llama-index-embeddings-huggingface

# !mkdir -p data
# !echo "This is a test file: one!" > data/test1.txt
# !echo "This is a test file: two!" > data/test2.txt


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp", filename_as_id=True).load_data()

"""
## Create Pipeline with Document Store
"""
logger.info("## Create Pipeline with Document Store")


pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    ],
    docstore=SimpleDocumentStore(),
)

nodes = pipeline.run(documents=documents)

logger.debug(f"Ingested {len(nodes)} Nodes")

"""
### [Optional] Save/Load Pipeline

Saving the pipeline will save both the internal cache and docstore.

**NOTE:** If you were using remote caches/docstores, this step is not needed
"""
logger.info("### [Optional] Save/Load Pipeline")

pipeline.persist("./pipeline_storage")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    ]
)

pipeline.load("./pipeline_storage")

"""
## Test the Document Management

Here, we can create a new document, as well as edit an existing document, to test the document management.

Both the new document and edited document will be ingested, while the unchanged document will be skipped
"""
logger.info("## Test the Document Management")

# !echo "This is a test file: three!" > data/test3.txt
# !echo "This is a NEW test file: one!" > data/test1.txt

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp", filename_as_id=True).load_data()

nodes = pipeline.run(documents=documents)

logger.debug(f"Ingested {len(nodes)} Nodes")

"""
Lets confirm which nodes were ingested:
"""
logger.info("Lets confirm which nodes were ingested:")

for node in nodes:
    logger.debug(f"Node: {node.text}")

"""
We can also verify the docstore has only three documents tracked
"""
logger.info("We can also verify the docstore has only three documents tracked")

logger.debug(len(pipeline.docstore.docs))

logger.info("\n\n[DONE]", bright=True)
