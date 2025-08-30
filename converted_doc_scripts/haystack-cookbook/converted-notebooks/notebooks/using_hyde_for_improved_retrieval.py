from datasets import load_dataset, Dataset
from haystack import Document
from haystack import Pipeline
from haystack import Pipeline, Document
from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack import component
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OllamaFunctionCallingAdapterGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from jet.logger import CustomLogger
from numpy import array, mean
from typing import Dict, Any, List
from typing import List
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
# Using Hypothetical Document Embeddings (HyDE) to Improve Retrieval

> 📚 This cookbook has an accompanying article with a complete walkthrough ["Optimizing Retrival with HyDE"](https://haystack.deepset.ai/blog/optimizing-retrieval-with-hyde)

In this coookbook, we are building Haystack components that allow us to easily incorporate HyDE into our RAG pipelines, to optimize retrieval.

> To learn more about HyDE and when it's useful, check out our [guide to Hypothetical Document Embeddings (HyDE)](https://docs.haystack.deepset.ai/v2.0/docs/hypothetical-document-embeddings-hyde)

## Install Requirements
"""
logger.info("# Using Hypothetical Document Embeddings (HyDE) to Improve Retrieval")

# !pip install haystack-ai sentence-transformers datasets

"""
In the following sections, we will be using the `OllamaFunctionCallingAdapterGenerator`, so we need to provide our API key 👇
"""
logger.info("In the following sections, we will be using the `OllamaFunctionCallingAdapterGenerator`, so we need to provide our API key 👇")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Enter your openAI key:")

"""
## Building a Pipeline for Hypothetical Document Embeddings

We will build a Haystack pipeline that generates 'fake' documents.
For this part, we are using the `OllamaFunctionCallingAdapterGenerator` with a `PromptBuilder` that instructs the model to generate paragraphs.
"""
logger.info("## Building a Pipeline for Hypothetical Document Embeddings")


generator = OllamaFunctionCallingAdapterGenerator(
    model="llama3.2",
    generation_kwargs={"n": 5, "temperature": 0.75, "max_tokens": 400},
)

template="""Given a question, generate a paragraph of text that answers the question.
            Question: {{question}}
            Paragraph:"""

prompt_builder = PromptBuilder(template=template)

"""
Next, we use the `OutputAdapter` to transform the generated paragraphs into a List of Documents. This way, we will be able to use the `SentenceTransformersDocumentEmbedder` to create embeddings, since this component expects `List[Document]`
"""
logger.info("Next, we use the `OutputAdapter` to transform the generated paragraphs into a List of Documents. This way, we will be able to use the `SentenceTransformersDocumentEmbedder` to create embeddings, since this component expects `List[Document]`")


adapter = OutputAdapter(
    template="{{answers | build_doc}}",
    output_type=List[Document],
    custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]}
)

embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()

"""
Finally, we create a custom component, `HypotheticalDocumentEmbedder`, that expects `documents` and can return a list of `hypotethetical_embeddings` which is the average of the embeddings from the "hypothetical" (fake) documents. To learn more about this technique and where it's useful, check out our [Guide to HyDE](https://docs.haystack.deepset.ai/v2.0/docs/hypothetical-document-embeddings-hyde)
"""
logger.info("Finally, we create a custom component, `HypotheticalDocumentEmbedder`, that expects `documents` and can return a list of `hypotethetical_embeddings` which is the average of the embeddings from the "hypothetical" (fake) documents. To learn more about this technique and where it's useful, check out our [Guide to HyDE](https://docs.haystack.deepset.ai/v2.0/docs/hypothetical-document-embeddings-hyde)")


@component
class HypotheticalDocumentEmbedder:

  @component.output_types(hypothetical_embedding=List[float])
  def run(self, documents: List[Document]):
    stacked_embeddings = array([doc.embedding for doc in documents])
    avg_embeddings = mean(stacked_embeddings, axis=0)
    hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
    return {"hypothetical_embedding": hyde_vector[0].tolist()}

"""
We add all of our components into a pipeline to genereate a hypothetical document embedding 🚀👇
"""
logger.info("We add all of our components into a pipeline to genereate a hypothetical document embedding 🚀👇")


hyde = HypotheticalDocumentEmbedder()

pipeline = Pipeline()
pipeline.add_component(name="prompt_builder", instance=prompt_builder)
pipeline.add_component(name="generator", instance=generator)
pipeline.add_component(name="adapter", instance=adapter)
pipeline.add_component(name="embedder", instance=embedder)
pipeline.add_component(name="hyde", instance=hyde)

pipeline.connect("prompt_builder", "generator")
pipeline.connect("generator.replies", "adapter.answers")
pipeline.connect("adapter.output", "embedder.documents")
pipeline.connect("embedder.documents", "hyde.documents")

query = "What should I do if I have a fever?"
result = pipeline.run(data={"prompt_builder": {"question": query}})

logger.debug(result["hyde"])

"""
## Build a HyDE Component That Encapsulates the Whole Logic

This section shows you how to create a `HypotheticalDocumentEmbedder` that instead, encapsulates the entire logic, and also allows us to provide the embedding model as an optional parameter.

This "mega" components does a few things:

- Allows the user to pick the LLM which generates the hypothetical documents
- Allows users to define how many documents should be created with `nr_completions`
- Allows users to define the embedding model they want to use to generate the HyDE embeddings.
"""
logger.info("## Build a HyDE Component That Encapsulates the Whole Logic")




@component
class HypotheticalDocumentEmbedder:

    def __init__(
        self,
        instruct_llm: str = "llama3.2",
#         instruct_llm_api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        nr_completions: int = 5,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.instruct_llm = instruct_llm
        self.instruct_llm_api_key = instruct_llm_api_key
        self.nr_completions = nr_completions
        self.embedder_model = embedder_model
        self.generator = OllamaFunctionCallingAdapterGenerator(
            api_key=self.instruct_llm_api_key,
            model=self.instruct_llm,
            generation_kwargs={"n": self.nr_completions, "temperature": 0.75, "max_tokens": 400},
        )
        self.prompt_builder = PromptBuilder(
            template="""Given a question, generate a paragraph of text that answers the question.
            Question: {{question}}
            Paragraph:
            """
        )

        self.adapter = OutputAdapter(
            template="{{answers | build_doc}}",
            output_type=List[Document],
            custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]},
        )

        self.embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, progress_bar=False)
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="adapter", instance=self.adapter)
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.answers")
        self.pipeline.connect("adapter.output", "embedder.documents")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            instruct_llm=self.instruct_llm,
            instruct_llm_api_key=self.instruct_llm_api_key,
            nr_completions=self.nr_completions,
            embedder_model=self.embedder_model,
        )
        data["pipeline"] = self.pipeline.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        hyde_obj = default_from_dict(cls, data)
        hyde_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyde_obj

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"question": query}})
        stacked_embeddings = array([doc.embedding for doc in result["embedder"]["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}

"""
## Use HyDE For Retrieval

Let's see how we can use this component in a full pipeline. First, let's index some documents into an `InMemoryDocumentStore`
"""
logger.info("## Use HyDE For Retrieval")



embedder_model = "sentence-transformers/all-MiniLM-L6-v2"


def index_docs(data: Dataset):
    document_store = InMemoryDocumentStore()

    pipeline = Pipeline()
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedder_model))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy="skip"))

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    pipeline.run({"cleaner": {"documents": [Document.from_dict(doc) for doc in data["train"]]}})
    return document_store


data = load_dataset("Tuana/game-of-thrones")
doc_store = index_docs(data)

"""
We can now run a retrieval pipeline that doesn't just retrieve based on the query embeddings, instead, it uses the `HypotheticalDocumentEmbedder` to create hypothetical document embeddings based on our `query` and uses these new embeddings to retrieve documents.
"""
logger.info("We can now run a retrieval pipeline that doesn't just retrieve based on the query embeddings, instead, it uses the `HypotheticalDocumentEmbedder` to create hypothetical document embeddings based on our `query` and uses these new embeddings to retrieve documents.")


def retriever_with_hyde(doc_store):
    hyde = HypotheticalDocumentEmbedder(instruct_llm="llama3.2", nr_completions=5)
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store)

    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component(instance=hyde, name="query_embedder")
    retrieval_pipeline.add_component(instance=retriever, name="retriever")

    retrieval_pipeline.connect("query_embedder.hypothetical_embedding", "retriever.query_embedding")
    return retrieval_pipeline

retrieval_pipeline = retriever_with_hyde(doc_store)
query = "Who is Araya Stark?"
retrieval_pipeline.run(data={"query_embedder": {"query": query}, "retriever": {"top_k": 5}})

logger.info("\n\n[DONE]", bright=True)