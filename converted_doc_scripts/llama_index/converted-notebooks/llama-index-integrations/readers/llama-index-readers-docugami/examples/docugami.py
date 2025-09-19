from base import DocugamiReader
from jet.logger import CustomLogger
from llama_index import VectorStoreIndex
from llama_index import download_loader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# %load_ext autoreload
# %autoreload 2

"""
# Docugami
This notebook covers how to load documents from `Docugami`. See [README](./README.md) for more details, and the advantages of using this system over alternative data loaders.

## Prerequisites
1. Follow the Quick Start section in [README](./README.md)
2. Grab an access token for your workspace, and make sure it is set as the DOCUGAMI_API_KEY environment variable
3. Grab some docset and document IDs for your processed documents, as described here: https://help.docugami.com/home/docugami-api

## Load Documents

If the DOCUGAMI_API_KEY environment variable is set, there is no need to pass it in to the loader explicitly otherwise you can pass it in as the `access_token` parameter.

The DocugamiReader has a default minimum chunk size of 32. Chunks smaller than that are appended to subsequent chunks. Set min_chunk_size to 0 to get all structural chunks regardless of size.
"""
logger.info("# Docugami")


docset_id = "ecxqpipcoe2p"
document_ids = ["43rj0ds7s0ur", "bpc1vibyeke2"]

loader = DocugamiReader()
documents = loader.load_data(docset_id=docset_id, document_ids=document_ids)

"""
The `metadata` for each `Document` (really, a chunk of an actual PDF, DOC or DOCX) contains some useful additional information:

1. **id and name:** ID and Name of the file (PDF, DOC or DOCX) the chunk is sourced from within Docugami.
2. **xpath:** XPath inside the XML representation of the document, for the chunk. Useful for source citations directly to the actual chunk inside the document XML.
3. **structure:** Structural attributes of the chunk, e.g. h1, h2, div, table, td, etc. Useful to filter out certain kinds of chunks if needed by the caller.
4. **tag:** Semantic tag for the chunk, using various generative and extractive techniques. More details here: https://github.com/docugami/DFM-benchmarks

## Basic Use: Docugami Loader for Document QA

You can use the Docugami Loader like a standard loader for Document QA over multiple docs, albeit with much better chunks that follow the natural contours of the document. There are many great tutorials on how to do this, e.g. [this one](https://gpt-index.readthedocs.io/en/latest/getting_started/starter_example.html). We can just use the same code, but use the `DocugamiLoader` for better chunking, instead of loading text or PDF files directly with basic splitting techniques.
"""
logger.info("## Basic Use: Docugami Loader for Document QA")


docset_id = "wh2kned25uqm"
documents = loader.load_data(docset_id=docset_id)

for d in documents:
    stripped_metadata = d.metadata.copy()
    for key in d.metadata:
        if key not in ["name", "xpath", "id", "structure"]:
            del stripped_metadata[key]
    d.metadata = stripped_metadata

documents

"""
The documents returned by the loader are already split into chunks. Optionally, we can use the metadata on each chunk, for example the structure or tag attributes, to do any post-processing we want.

We will just use the output of the `DocugamiLoader` as-is to set up a query engine the usual way.
"""
logger.info("The documents returned by the loader are already split into chunks. Optionally, we can use the metadata on each chunk, for example the structure or tag attributes, to do any post-processing we want.")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("What can tenants do with signage on their properties?")
logger.debug(response.response)
for node in response.source_nodes:
    logger.debug(node)

"""
## Using Docugami to Add Metadata to Chunks for High Accuracy Document QA

One issue with large documents is that the correct answer to your question may depend on chunks that are far apart in the document. Typical chunking techniques, even with overlap, will struggle with providing the LLM sufficientcontext to answer such questions. With upcoming very large context LLMs, it may be possible to stuff a lot of tokens, perhaps even entire documents, inside the context but this will still hit limits at some point with very long documents, or a lot of documents.

For example, if we ask a more complex question that requires the LLM to draw on chunks from different parts of the document, even OllamaFunctionCalling's powerful LLM is unable to answer correctly.
"""
logger.info("## Using Docugami to Add Metadata to Chunks for High Accuracy Document QA")

response = query_engine.query(
    "What is the security deposit for the property owned by Birch Street?"
)
logger.debug(response.response)  # the correct answer should be $78,000
for node in response.source_nodes:
    logger.debug(node.node.extra_info["name"])
    logger.debug(node.node.text)

"""
At first glance the answer may seem plausible, but if you review the source chunks carefully for this answer, you will see that the chunking of the document did not end up putting the Landlord name and the rentable area in the same context, since they are far apart in the document. The query engine therefore ends up finding unrelated chunks from other documents not even related to the **Birch Street** landlord. That landlord happens to be mentioned on the first page of the file **TruTone Lane 1.docx** file, and none of the source chunks used by the query engine contain the correct answer (**$78,000**), and the answer is therefore incorrect.

Docugami can help here. Chunks are annotated with additional metadata created using different techniques if a user has been [using Docugami](https://help.docugami.com/home/reports). More technical approaches will be added later.

Specifically, let's load the data again and this time instead of stripping semantic metadata let's look at the additional metadata that is returned on the documents returned by docugami after some additional use, in the form of some simple key/value pairs on all the text chunks:
"""
logger.info("At first glance the answer may seem plausible, but if you review the source chunks carefully for this answer, you will see that the chunking of the document did not end up putting the Landlord name and the rentable area in the same context, since they are far apart in the document. The query engine therefore ends up finding unrelated chunks from other documents not even related to the **Birch Street** landlord. That landlord happens to be mentioned on the first page of the file **TruTone Lane 1.docx** file, and none of the source chunks used by the query engine contain the correct answer (**$78,000**), and the answer is therefore incorrect.")

docset_id = "wh2kned25uqm"
documents = loader.load_data(docset_id=docset_id)
documents[0].metadata

"""
Note semantic metadata tags like Lease Date, Landlord, Tenant, etc that are based on key chunks in the document even if they don't appear near the chunk in question.
"""
logger.info("Note semantic metadata tags like Lease Date, Landlord, Tenant, etc that are based on key chunks in the document even if they don't appear near the chunk in question.")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

"""
Let's run the same question again. It returns the correct result since all the chunks have metadata key/value pairs on them carrying key information about the document even if this information is physically very far away from the source chunk used to generate the answer.
"""
logger.info("Let's run the same question again. It returns the correct result since all the chunks have metadata key/value pairs on them carrying key information about the document even if this information is physically very far away from the source chunk used to generate the answer.")

response = query_engine.query(
    "What is the security deposit for the property owned by Birch Street?"
)
logger.debug(response.response)  # the correct answer should be $78,000
for node in response.source_nodes:
    logger.debug(node.node.extra_info["name"])
    logger.debug(node.node.text)

logger.info("\n\n[DONE]", bright=True)