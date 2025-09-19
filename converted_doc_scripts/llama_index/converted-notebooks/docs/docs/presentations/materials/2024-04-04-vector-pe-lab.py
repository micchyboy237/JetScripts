from jet.models.config import MODELS_CACHE_DIR
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import List
import json
import os
import qdrant_client
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LLMs and LlamaIndex ◦ April 4 2024 ◦ Vector Institute Prompt Engineering Lab
"""
logger.info(
    "# LLMs and LlamaIndex ◦ April 4 2024 ◦ Vector Institute Prompt Engineering Lab")


"""
![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/title.excalidraw.svg)

![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/framework.excalidraw.svg)

#### Notebook Setup & Dependency Installation
"""
logger.info("#### Notebook Setup & Dependency Installation")

# %pip install llama-index-vector-stores-qdrant -q

# import nest_asyncio

# nest_asyncio.apply()

# !mkdir data
# !wget "https://vectorinstitute.ai/wp-content/uploads/2024/02/Vector-Annual-Report-2022-23_accessible_rev0224-1.pdf" -O "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/Vector-Annual-Report-2022-23_accessible_rev0224-1.pdf"

"""
## Motivation

![Motivation Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/motivation.excalidraw.svg)
"""
logger.info("## Motivation")


llm = OllamaFunctionCalling(
    model="llama3.2")

response = llm.complete("What is Vector Institute all about?")

logger.debug(response)

response = llm.complete(
    "According to Vector Institute's Annual Report 2022-2023, "
    "how many AI jobs were created in Ontario?"
)

logger.debug(response)

"""
## Basic RAG in 3 Steps

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/subheading.excalidraw.svg)


1. Build external knowledge (i.e., uploading updated data sources)
2. Retrieve
3. Augment and Generate

## 1. Build External Knowledge

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step1.excalidraw.svg)
"""
logger.info("## Basic RAG in 3 Steps")

"""Load the data.

With llama-index, before any transformations are applied,
data is loaded in the `Document` abstraction, which is
a container that holds the text of the document.
"""


loader = SimpleDirectoryReader(
    input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp")
documents = loader.load_data()

documents[1].text

"""Chunk, Encode, and Store into a Vector Store.

To streamline the process, we can make use of the IngestionPipeline
class that will apply your specified transformations to the
Document's.
"""


client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    ],
    vector_store=vector_store,
)
_nodes = pipeline.run(documents=documents, num_workers=4)

_nodes[1].text

"""Create a llama-index... wait for it... Index.

After uploading your encoded documents into your vector
store of choice, you can connect to it with a VectorStoreIndex
which then gives you access to all of the llama-index functionality.
"""


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
## 2. Retrieve Against A Query

![Step2 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step2.excalidraw.svg)
"""
logger.info("## 2. Retrieve Against A Query")

"""Retrieve relevant documents against a query.

With our Index ready, we can now query it to
retrieve the most relevant document chunks.
"""

retriever = index.as_retriever(similarity_top_k=2)
retrieved_nodes = retriever.retrieve(
    "According to Vector Institute's Annual Report 2022-2023, "
    "how many AI jobs were created in Ontario?"
)

retrieved_nodes

"""
## 3. Generate Final Response

![Step3 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step3.excalidraw.svg)
"""
logger.info("## 3. Generate Final Response")

"""Context-Augemented Generation.

With our Index ready, we can create a QueryEngine
that handles the retrieval and context augmentation
in order to get the final response.
"""

query_engine = index.as_query_engine()

logger.debug(
    query_engine.get_prompts()[
        "response_synthesizer:text_qa_template"
    ].default_template.template
)

response = query_engine.query(
    "According to Vector Institute's Annual Report 2022-2023, "
    "how many AI jobs were created in Ontario?"
)
logger.debug(response)

"""
## More Queries

### Comparisons
"""
logger.info("## More Queries")

query = (
    "According to Vector Institute's 2022-2023 annual report, "
    "how many new AI companies were established in Ontario?"
)

response = query_engine.query(query)
logger.debug(response)

query = (
    "According to Vector Institute's 2022-2023 annual report, "
    "what was the dollar value for Unrestricted net assets in "
    "the years 2022 & 2023?"
)

response = query_engine.query(query)
logger.debug(response)

query = (
    "According to Vector Institute's 2022-2023 annual report, "
    "what companies were platinum sponsors?"
)

response = query_engine.query(query)
logger.debug(response)

"""
## In Summary

- LLMs as powerful as they are, don't perform too well with knowledge-intensive tasks (domain specific, updated data, long-tail)
- Context augmentation has been shown (in a few studies) to outperform LLMs without augmentation
- In this notebook, we showed one such example that follows that pattern.

## Storing Results In Structured Objects

![DataExtractions](https://media.licdn.com/dms/image/D4E22AQGwPmZ5RRhbyA/feedshare-shrink_1280/0/1711823067172?e=1715212800&v=beta&t=fJtksPZ3Fm-BOrKRCwa6BrYyuxlNFDuop3ZSopMl53M)
"""
logger.info("## In Summary")


"""
### Sponsors
"""
logger.info("### Sponsors")


class VectorSponsors(BaseModel):
    """Data model for Vector Institute sponsors 2022-2023."""

    platinum: str = Field(description="Platinum sponsors")
    gold: str = Field(description="Gold sponsors")
    silver: str = Field(description="Silver sponsors")
    bronze: List[str] = Field(description="Bronze sponsors")


prompt_template_str = """\
Here is the 2022-2023 Annual Report for Vector Institute:
{document_text}
Provide the names sponsor companies according to their tiers.
"""

program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=VectorSponsors,
    prompt_template_str=prompt_template_str,
    llm=OllamaFunctionCalling("gpt-4-turbo-preview"),
    verbose=True,
)

report_as_text = ""
for d in documents:
    report_as_text += d.text

sponsors = program(document_text=report_as_text)

logger.debug(json.dumps(sponsors.dict(), indent=4))

"""
### Useful links

[website](https://www.llamaindex.ai/) ◦ [llamahub](https://llamahub.ai) ◦ [llamaparse](https://github.com/run-llama/llama_parse) ◦ [github](https://github.com/run-llama/llama_index) ◦ [medium](https://medium.com/@llama_index) ◦ [rag-bootcamp-poster](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/final_poster.excalidraw.svg)
"""
logger.info("### Useful links")

logger.info("\n\n[DONE]", bright=True)
