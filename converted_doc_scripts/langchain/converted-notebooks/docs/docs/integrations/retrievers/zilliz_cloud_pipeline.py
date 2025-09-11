from jet.logger import logger
from langchain_milvus import ZillizCloudPipelineRetriever
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
# Zilliz Cloud Pipeline

> [Zilliz Cloud Pipelines](https://docs.zilliz.com/docs/pipelines) transform your unstructured data to a searchable vector collection, chaining up the embedding, ingestion, search, and deletion of your data.
> 
> Zilliz Cloud Pipelines are available in the Zilliz Cloud Console and via RestFul APIs.

This notebook demonstrates how to prepare Zilliz Cloud Pipelines and use the them via a LangChain Retriever.

## Prepare Zilliz Cloud Pipelines

To get pipelines ready for LangChain Retriever, you need to create and configure the services in Zilliz Cloud.

**1. Set up Database**

- [Register with Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud)
- [Create a cluster](https://docs.zilliz.com/docs/create-cluster)

**2. Create Pipelines**

- [Document ingestion, search, deletion](https://docs.zilliz.com/docs/pipelines-doc-data)
- [Text ingestion, search, deletion](https://docs.zilliz.com/docs/pipelines-text-data)

## Use LangChain Retriever
"""
logger.info("# Zilliz Cloud Pipeline")

# %pip install --upgrade --quiet langchain-milvus


retriever = ZillizCloudPipelineRetriever(
    pipeline_ids={
        "ingestion": "<YOUR_INGESTION_PIPELINE_ID>",  # skip this line if you do NOT need to add documents
        "search": "<YOUR_SEARCH_PIPELINE_ID>",  # skip this line if you do NOT need to get relevant documents
        "deletion": "<YOUR_DELETION_PIPELINE_ID>",  # skip this line if you do NOT need to delete documents
    },
    token="<YOUR_ZILLIZ_CLOUD_API_KEY>",
)

"""
### Add documents

To add documents, you can use the method `add_texts` or `add_doc_url`, which inserts documents from a list of texts or a presigned/public url with corresponding metadata into the store.

- if using a **text ingestion pipeline**, you can use the method `add_texts`, which inserts a batch of texts with the corresponding metadata into the Zilliz Cloud storage.

    **Arguments:**
    - `texts`: A list of text strings.
    - `metadata`: A key-value dictionary of metadata will be inserted as preserved fields required by ingestion pipeline. Defaults to None.
"""
logger.info("### Add documents")



"""
- if using a **document ingestion pipeline**, you can use the method `add_doc_url`, which inserts a document from url with the corresponding metadata into the Zilliz Cloud storage.

    **Arguments:**
    - `doc_url`: A document url.
    - `metadata`: A key-value dictionary of metadata will be inserted as preserved fields required by ingestion pipeline. Defaults to None.

The following example works with a document ingestion pipeline, which requires milvus version as metadata. We will use an [example document](https://publicdataset.zillizcloud.com/milvus_doc.md) describing how to delete entities in Milvus v2.3.x.
"""
logger.info("The following example works with a document ingestion pipeline, which requires milvus version as metadata. We will use an [example document](https://publicdataset.zillizcloud.com/milvus_doc.md) describing how to delete entities in Milvus v2.3.x.")

retriever.add_doc_url(
    doc_url="https://publicdataset.zillizcloud.com/milvus_doc.md",
    metadata={"version": "v2.3.x"},
)

"""
### Get relevant documents

To query the retriever, you can use the method `get_relevant_documents`, which returns a list of LangChain Document objects.

**Arguments:**
- `query`: String to find relevant documents for.
- `top_k`: The number of results. Defaults to 10.
- `offset`: The number of records to skip in the search result. Defaults to 0.
- `output_fields`: The extra fields to present in output.
- `filter`: The Milvus expression to filter search results. Defaults to "".
- `run_manager`: The callbacks handler to use.
"""
logger.info("### Get relevant documents")

retriever.get_relevant_documents(
    "Can users delete entities by complex boolean expressions?"
)

"""

"""

logger.info("\n\n[DONE]", bright=True)