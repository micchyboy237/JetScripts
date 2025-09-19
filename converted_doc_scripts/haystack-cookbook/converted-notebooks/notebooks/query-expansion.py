from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_experimental.components.query.query_expander import QueryExpander
from jet.logger import CustomLogger
from typing import List, Optional
import json
import os
import shutil
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Advanced RAG: Query Expansion
_by Tuana Celik ([LI](https://www.linkedin.com/in/tuanacelik/),  [Twitter/X](https://x.com/tuanacelik))_

> This is part one of the **Advanced Use Cases** series:
>
> 1️⃣ Extract Metadata from Queries to Improve Retrieval [cookbook](/cookbook/extracting_metadata_filters_from_a_user_query) & [full article](/blog/extracting-metadata-filter)
>
> 2️⃣ **Query Expansion & the [full article](/blog/query-expansion)**
>
> 3️⃣ Query Decomposition [cookbook](/cookbook/query_decomposition) & [full article](/blog/query-decomposition)
>
> 4️⃣ [Automated Metadata Enrichment](/cookbook/metadata_enrichment)

In this cookbook, you'll learn how to implement query expansion for RAG. Query expansion consists of asking an LLM to produce a number of similar queries to a user query. We are then able to use each of these queries in the retrieval process, increasing the number and relevance of retrieved documents.

📚 [Read the full article](https://haystack.deepset.ai/blog/query-expansion)
"""
logger.info("# Advanced RAG: Query Expansion")

# !pip install haystack-ai wikipedia






# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#   os.environ['OPENAI_API_KEY'] = getpass("Your OllamaFunctionCalling API Key: ")

"""
## The Process of Query Expansion

First, let's import the `QueryExpander` from **Haystack Experimental**.

Next, we’ll create a `QueryExpander` instance. This component generates a specified number (default is 4) of additional queries that are similar to the original user query. It returns `queries`, which include the original query plus the generated similar ones.
"""
logger.info("## The Process of Query Expansion")


expander = QueryExpander()
expander.run(query="open source nlp frameworks", n_expansions=4)

"""
## Retrieval Without Query Expansion
"""
logger.info("## Retrieval Without Query Expansion")

documents = [
    Document(content="The effects of climate are many including loss of biodiversity"),
    Document(content="The impact of climate change is evident in the melting of the polar ice caps."),
    Document(content="Consequences of global warming include the rise in sea levels."),
    Document(content="One of the effects of environmental changes is the change in weather patterns."),
    Document(content="There is a global call to reduce the amount of air travel people take."),
    Document(content="Air travel is one of the core contributors to climate change."),
    Document(content="Expect warm climates in Turkey during the summer period."),
]

doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
doc_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store=doc_store, top_k=3)

retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("keyword_retriever", retriever)

query = "climate change"
retrieval_pipeline.run({"keyword_retriever":{ "query": query, "top_k": 3}})

"""
## Retrieval With Query Expansion

Now let's have a look at what documents we are able to retrieve if we are to inluce query expansion in the process. For this step, let's create a `MultiQueryInMemoryBM25Retriever` that is able to use BM25 retrieval for each (expansded) query in turn.

This component also handles the same document being retrieved for multiple queries and will not return duplicates.
"""
logger.info("## Retrieval With Query Expansion")

@component
class MultiQueryInMemoryBM25Retriever:

    def __init__(self, retriever: InMemoryBM25Retriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = None):
        if top_k != None:
            self.top_k = top_k

        all_docs = {}
        for query in queries:
            result = self.retriever.run(query = query, top_k = self.top_k)
            for doc in result['documents']:
                all_docs[doc.id] = doc

        all_docs = list(all_docs.values())
        all_docs.sort(key=lambda x: x.score, reverse=True)
        return {"documents": all_docs}

query_expander = QueryExpander()
retriever = MultiQueryInMemoryBM25Retriever(InMemoryBM25Retriever(document_store=doc_store))

expanded_retrieval_pipeline = Pipeline()
expanded_retrieval_pipeline.add_component("expander", query_expander)
expanded_retrieval_pipeline.add_component("keyword_retriever", retriever)

expanded_retrieval_pipeline.connect("expander.queries", "keyword_retriever.queries")

expanded_retrieval_pipeline.run({"expander": {"query": query}}, include_outputs_from=["expander"])

"""
## Query Expansion for RAG

Let's start off by populating a document store with chunks of context from various Wikipedia pages.
"""
logger.info("## Query Expansion for RAG")

def get_doc_store():
    raw_docs = []
    wikipedia_page_titles = ["Electric_vehicle", "Dam", "Electric_battery", "Tree", "Solar_panel", "Nuclear_power",
                             "Wind_power", "Hydroelectricity", "Coal", "Natural_gas", "Greenhouse_gas", "Renewable_energy",
                             "Fossil_fuel"]
    for title in wikipedia_page_titles:
        page = wikipedia.page(title=title, auto_suggest=False)
        doc = Document(content=page.content, meta={"title": page.title, "url": page.url})
        raw_docs.append(doc)

    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="passage", split_length=1))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP))
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")

    indexing_pipeline.run({"cleaner": {"documents": raw_docs}})

    return doc_store

doc_store = get_doc_store()

"""
### RAG without Query Expansion
"""
logger.info("### RAG without Query Expansion")

chat_message = ChatMessage.from_user(
    text="""You are part of an information system that summarises related documents.
You answer a query using the textual content from the documents retrieved for the
following query.
You build the summary answer based only on quoting information from the documents.
You should reference the documents you used to support your answer.
Original Query: "{{query}}"
Retrieved Documents: {{documents}}
Summary Answer:
"""
)
retriever = InMemoryBM25Retriever(document_store=doc_store)
chat_prompt_builder = ChatPromptBuilder(template=[chat_message], required_variables="*")
llm = OllamaFunctionCallingAdapterChatGenerator()

keyword_rag_pipeline = Pipeline()
keyword_rag_pipeline.add_component("keyword_retriever", retriever)
keyword_rag_pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
keyword_rag_pipeline.add_component("llm", llm)

keyword_rag_pipeline.connect("keyword_retriever.documents", "chat_prompt_builder.documents")
keyword_rag_pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

keyword_rag_pipeline.run({"query": "green energy sources", "top_k": 3}, include_outputs_from=["keyword_retriever"])

"""
### RAG with Query Expansion
"""
logger.info("### RAG with Query Expansion")

chat_message = ChatMessage.from_user(
    text="""You are part of an information system that summarises related documents.
You answer a query using the textual content from the documents retrieved for the
following query.
You build the summary answer based only on quoting information from the documents.
You should reference the documents you used to support your answer.
Original Query: "{{query}}"
Retrieved Documents: {{documents}}
Summary Answer:
"""
)
query_expander = QueryExpander()
retriever = MultiQueryInMemoryBM25Retriever(InMemoryBM25Retriever(document_store=doc_store))
chat_prompt_builder = ChatPromptBuilder(template=[chat_message], required_variables="*")
llm = OllamaFunctionCallingAdapterChatGenerator()

query_expanded_rag_pipeline = Pipeline()
query_expanded_rag_pipeline.add_component("expander", query_expander)
query_expanded_rag_pipeline.add_component("keyword_retriever", retriever)
query_expanded_rag_pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
query_expanded_rag_pipeline.add_component("llm", llm)

query_expanded_rag_pipeline.connect("expander.queries", "keyword_retriever.queries")
query_expanded_rag_pipeline.connect("keyword_retriever.documents", "chat_prompt_builder.documents")
query_expanded_rag_pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

query_expanded_rag_pipeline.show()

query_expanded_rag_pipeline.run({"query": "green energy sources", "top_k": 3}, include_outputs_from=["keyword_retriever", "expander"])

logger.info("\n\n[DONE]", bright=True)