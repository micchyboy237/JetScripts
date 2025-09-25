from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder
from haystack_integrations.components.embedders.jina import JinaTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from jet.logger import logger
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
# Using the Jina-embeddings-v2-base-en model in a Haystack RAG pipeline for legal document analysis

One foggy day in October 2023, I was narrowly excused from jury duty. I had mixed feelings about it, since it actually seemed like a pretty interesting case (Google v. Sonos). A few months later, I idly wondered how the proceedings turned out. I could just read the news, but what's the fun in that? Let's see how AI can solve this problem.

[Jina.ai](https://jina.ai/) recently released `jina-embeddings-v2-base-en`. It's an open-source text embedding model capable of accommodating up to 8192 tokens. Splitting text into larger chunks is helpful for understanding longer documents. One of the use cases this model is especially suited for is legal document analysis.

In this demo, we'll build a [RAG pipeline](https://www.deepset.ai/blog/llms-retrieval-augmentation) to discover the outcome of the Google v. Sonos case, using the following technologies:
- the [`jina-embeddings-v2-base-en`](https://arxiv.org/abs/2310.19923) model
- [Haystack](https://haystack.deepset.ai/), the open source LLM orchestration framework
- [Chroma](https://docs.trychroma.com/getting-started) to store our vector embeddings, via the [Chroma Document Store Haystack integration](https://haystack.deepset.ai/integrations/chroma-documentstore)
- the open source [Mistral 7B Instruct LLM](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)


## Prerequisites:
- You need a Jina AI key - [get a free one here](https://jina.ai/embeddings/).
- You also need an [Hugging Face access token](https://huggingface.co/docs/hub/security-tokens)

First, install all our required dependencies.
"""
logger.info("# Using the Jina-embeddings-v2-base-en model in a Haystack RAG pipeline for legal document analysis")

# !pip3 install pypdf

# !pip install haystack-ai jina-haystack chroma-haystack "huggingface_hub>=0.22.0"

"""
Then input our credentials.
"""
logger.info("Then input our credentials.")

# from getpass import getpass

# os.environ["JINA_API_KEY"]  = getpass("JINA api key:")
# os.environ["HF_API_TOKEN"] = getpass("Enter your HuggingFace api token: ")

"""
## Build an Indexing Pipeline

At a high level, the `LinkContentFetcher` pulls this document from its URL. Then we convert it from a PDF into a Document object Haystack can understand.

We preprocess it by removing whitespace and redundant substrings. Then split it into chunks, generate embeddings, and write these embeddings into the `ChromaDocumentStore`.
"""
logger.info("## Build an Indexing Pipeline")

document_store = ChromaDocumentStore()




fetcher = LinkContentFetcher()
converter = PyPDFToDocument()
cleaner = DocumentCleaner(remove_repeated_substrings=True)

splitter = DocumentSplitter(split_by="word", split_length=500)

writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

retriever = ChromaEmbeddingRetriever(document_store=document_store)

document_embedder = JinaDocumentEmbedder(model="jina-embeddings-v2-base-en")

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=fetcher, name="fetcher")
indexing_pipeline.add_component(instance=converter, name="converter")
indexing_pipeline.add_component(instance=cleaner, name="cleaner")
indexing_pipeline.add_component(instance=splitter, name="splitter")
indexing_pipeline.add_component(instance=document_embedder, name="embedder")
indexing_pipeline.add_component(instance=writer, name="writer")

indexing_pipeline.connect("fetcher.streams", "converter.sources")
indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

urls = ["https://cases.justia.com/federal/district-courts/california/candce/3:2020cv06754/366520/813/0.pdf"]

indexing_pipeline.run(data={"fetcher": {"urls": urls}})

"""
# Query pipeline

Now the real fun begins. Let's create a query pipeline so we can actually start asking questions. We write a prompt allowing us to pass our documents to the Mistral-7B LLM. Then we initiatialize the LLM via the `HuggingFaceAPIGenerator`.

To use this model, you need to accept the conditions here: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

In Haystack 2.0 `retriever`s are tightly coupled to `DocumentStores`. If we pass in the `retriever` we initialized earlier, this pipeline can access those embeddings we generated, and pass them to the LLM.
"""
logger.info("# Query pipeline")


prompt = """ Answer the question, based on the
content in the documents. If you can't answer based on the documents, say so.

Documents:
{% for doc in documents %}
  {{doc.content}}
{% endfor %}

question: {{question}}
"""

text_embedder = JinaTextEmbedder(model="jina-embeddings-v2-base-en")
generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"})


prompt_builder = PromptBuilder(template=prompt)
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder",text_embedder)
query_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("generator", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "generator.prompt")

"""
Time to ask a question!
"""
logger.info("Time to ask a question!")

question = "Summarize what happened in Google v. Sonos"

result = query_pipeline.run(data={"text_embedder":{"text": question},
                                  "retriever": {"top_k": 3},
                                  "prompt_builder":{"question": question},
                                  "generator": {"generation_kwargs": {"max_new_tokens": 350}}})

logger.debug(result['generator']['replies'][0])

"""
### Other questions you could try:
- What role did If This Then That play in Google v. Sonos?
- What judge presided over Google v. Sonos?
- What should Sonos have done differently?


### Alternate cases to explore
The indexing pipeline is written so that you can swap in other documents and analyze them. can You can try plugging the following URLs (or any PDF written in English) into the indexing pipeline and re-running all the code blocks below it.
- Google v. Oracle: https://supreme.justia.com/cases/federal/us/593/18-956/case.pdf
- JACK DANIEL’S PROPERTIES, INC. v. VIP PRODUCTS
LLC: https://www.supremecourt.gov/opinions/22pdf/22-148_3e04.pdf

Note: if you want to change the prompt template, you'll also need to re-run the code blocks starting where the `DocumentStore` is defined.

### Wrapping it up
Thanks for reading! If you're interested in learning more about the technologies used here, check out these blog posts:
- [Embeddings in Depth](https://jina.ai/news/embeddings-in-depth/)
- [What is text vectorization in NLP?](https://haystack.deepset.ai/blog/what-is-text-vectorization-in-nlp)
- [The definitive guide to BERT models](https://haystack.deepset.ai/blog/the-definitive-guide-to-bertmodels)
"""
logger.info("### Other questions you could try:")

logger.info("\n\n[DONE]", bright=True)