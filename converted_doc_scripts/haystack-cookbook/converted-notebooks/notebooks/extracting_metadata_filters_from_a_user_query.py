from haystack import Document
from haystack import Pipeline, Document
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from jet.logger import CustomLogger
from typing import Dict, List
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Extract Metadata Filters from a Query

*Notebook by [David Batista](https://www.linkedin.com/in/dsbatista)*

> This is part one of the **Advanced Use Cases** series:
>
> 1️⃣ **Extract Metadata from Queries to Improve Retrieval & the [full article](/blog/extracting-metadata-filter)**
>
> 2️⃣ Query Expansion [cookbook](/cookbook/query-expansion) & [full article](/blog/query-expansion)
>
> 3️⃣ Query Decomposition [cookbook](/cookbook/query_decomposition) & the [full article](/blog/query-decomposition)
>
> 4️⃣ [Automated Metadata Enrichment](/cookbook/metadata_enrichment)

In this notebook, we'll discuss how to implement a custom component, `QueryMetadataExtractor`, that extracts entities from the query and formulates the corresponding metadata filter.

**Useful Sources**

* [📖 Docs](https://docs.haystack.deepset.ai/docs/intro)
* [📚 Tutorials](https://haystack.deepset.ai/tutorials)
* [🧑‍🍳 Cookbooks](https://github.com/deepset-ai/haystack-cookbook)

## Setup the Development Environment
"""
logger.info("# Extract Metadata Filters from a Query")

# !pip install haystack-ai
# !pip install sentence-transformers

"""
# Enter your `OPENAI_API_KEY`. Get your OllamaFunctionCalling API key [here](https://platform.openai.com/api-keys):
"""
# logger.info("Enter your `OPENAI_API_KEY`. Get your OllamaFunctionCalling API key [here](https://platform.openai.com/api-keys):")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCalling API key:")

"""
## Implement `QueryMetadataExtractor`

Create a [custom component](https://docs.haystack.deepset.ai/docs/custom-components), `QueryMetadataExtractor`, which takes `query` and `metadata_fields` as inputs and outputs `filters`. This component encapsulates a generative pipeline, made up of [`PromptBuilder`](https://docs.haystack.deepset.ai/docs/promptbuilder) and [`OllamaFunctionCallingAdapterGenerator`](https://docs.haystack.deepset.ai/docs/openaigenerator). The pipeline instructs the LLM to extract keywords, phrases, or entities from a given query which can then be used as metadata filters. In the prompt, we include instructions to ensure the output format is in JSON and provide `metadata_fields` along with the `query` to ensure the correct entities are extracted from the query.

Once the pipeline is initialized in the `init` method of the component, we post-process the LLM output in the `run` method. This step ensures the extracted metadata is correctly formatted to be used as a metadata filter.
"""
logger.info("## Implement `QueryMetadataExtractor`")



@component()
class QueryMetadataExtractor:

    def __init__(self):
        prompt = """
        You are part of an information system that processes users queries.
        Given a user query you extract information from it that matches a given list of metadata fields.
        The information to be extracted from the query must match the semantics associated with the given metadata fields.
        The information that you extracted from the query will then be used as filters to narrow down the search space
        when querying an index.
        Just include the value of the extracted metadata without including the name of the metadata field.
        The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.
        Example 1:
        Query: "What was the revenue of Nvidia in 2022?"
        Metadata fields: {"company", "year"}
        Extracted metadata fields: {"company": "nvidia", "year": 2022}
        Example 2:
        Query: "What were the most influential publications in 2023 regarding Alzheimer's disease?"
        Metadata fields: {"disease", "year"}
        Extracted metadata fields: {"disease": "Alzheimer", "year": 2023}
        Example 3:
        Query: "{{query}}"
        Metadata fields: "{{metadata_fields}}"
        Extracted metadata fields:
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(prompt))
        self.pipeline.add_component(name="llm", instance=OllamaFunctionCallingAdapterGenerator(model="llama3.2"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(filters=Dict[str, str])
    def run(self, query: str, metadata_fields: List[str]):
        result = self.pipeline.run({'builder': {'query': query, 'metadata_fields': metadata_fields}})
        metadata = json.loads(result['llm']['replies'][0])

        filters = []
        for key, value in metadata.items():
            field = f"meta.{key}"
            filters.append({f"field": field, "operator": "==", "value": value})

        return {"filters": {"operator": "AND", "conditions": filters}}

"""
First, let's test the `QueryMetadataExtractor` in isolation, passing a query and a list of metadata fields.
"""
logger.info("First, let's test the `QueryMetadataExtractor` in isolation, passing a query and a list of metadata fields.")

extractor = QueryMetadataExtractor()

query = "What were the most influential publications in 2022 regarding Parkinson's disease?"
metadata_fields = {"disease", "year"}

result = extractor.run(query, metadata_fields)
logger.debug(result)

"""
Notice that the `QueryMetadataExtractor` has extracted the metadata fields from the query and returned them in a format that can be used as filters passed directly to a `Retriever`. By default, the `QueryMetadataExtractor` will use all metadata fields as conditions together with an `AND` operator.

## Use `QueryMetadataExtractor` in a Pipeline

Now, let's plug the `QueryMetadataExtractor` into a `Pipeline` with a `Retriever` connected to a `DocumentStore` to see how it works in practice.

We start by creating a `InMemoryDocumentStore` and adding some documents to it. We include info about “year” and “disease” in the “meta” field of each document.
"""
logger.info("## Use `QueryMetadataExtractor` in a Pipeline")


documents = [
    Document(
        content="some publication about Alzheimer prevention research done over 2023 patients study",
        meta={"year": 2022, "disease": "Alzheimer", "author": "Michael Butter"}),
    Document(
        content="some text about investigation and treatment of Alzheimer disease",
        meta={"year": 2023, "disease": "Alzheimer", "author": "John Bread"}),
    Document(
        content="A study on the effectiveness of new therapies for Parkinson's disease",
        meta={"year": 2022, "disease": "Parkinson", "author": "Alice Smith"}
    ),
    Document(
        content="An overview of the latest research on the genetics of Parkinson's disease and its implications for treatment",
        meta={"year": 2023, "disease": "Parkinson", "author": "David Jones"}
    )
]

document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
document_store.write_documents(documents=documents, policy=DuplicatePolicy.OVERWRITE)

"""
We then create a pipeline consisting of the `QueryMetadataExtractor` and a `InMemoryBM25Retriever` connected to the `InMemoryDocumentStore` created above.

> Learn about connecting components and creating pipelines in [Docs: Creating Pipelines](https://docs.haystack.deepset.ai/docs/creating-pipelines).
"""
logger.info("We then create a pipeline consisting of the `QueryMetadataExtractor` and a `InMemoryBM25Retriever` connected to the `InMemoryDocumentStore` created above.")



retrieval_pipeline = Pipeline()
metadata_extractor = QueryMetadataExtractor()
retriever = InMemoryBM25Retriever(document_store=document_store)

retrieval_pipeline.add_component(instance=metadata_extractor, name="metadata_extractor")
retrieval_pipeline.add_component(instance=retriever, name="retriever")
retrieval_pipeline.connect("metadata_extractor.filters", "retriever.filters")

"""
Now define a query and metadata fields and pass them to the pipeline:
"""
logger.info("Now define a query and metadata fields and pass them to the pipeline:")

query = "publications 2023 Alzheimer's disease"
metadata_fields = {"year", "author", "disease"}

retrieval_pipeline.run(data={"metadata_extractor": {"query": query, "metadata_fields": metadata_fields}, "retriever":{"query": query}})

logger.info("\n\n[DONE]", bright=True)