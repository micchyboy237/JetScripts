from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from jet.logger import logger
from pprint import pprint
import os
import shutil
import wikipedia


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
# RAG pipeline using FastEmbed for embeddings generation

[FastEmbed](https://qdrant.github.io/fastembed/) is a lightweight, fast, Python library built for embedding generation, maintained by Qdrant. 
It is suitable for generating embeddings efficiently and fast on CPU-only machines.

In this notebook, we will use FastEmbed-Haystack integration to generate embeddings for indexing and RAG.

**Haystack Useful Sources**

* [Docs](https://docs.haystack.deepset.ai/docs/intro)
* [Tutorials](https://haystack.deepset.ai/tutorials)
* [Other Cookbooks](https://github.com/deepset-ai/haystack-cookbook)

## Install dependencies
"""
logger.info("# RAG pipeline using FastEmbed for embeddings generation")

# !pip install fastembed-haystack qdrant-haystack wikipedia transformers

"""
## Download contents and create docs
"""
logger.info("## Download contents and create docs")

favourite_bands="""Audioslave
Green Day
Muse (band)
Foo Fighters (band)
Nirvana (band)""".split("\n")


raw_docs=[]
for title in favourite_bands:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
## Clean, split and index documents on Qdrant
"""
logger.info("## Clean, split and index documents on Qdrant")


document_store = QdrantDocumentStore(
    ":memory:",
    embedding_dim =384,
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,
)

cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by='period', split_length=3)
splitted_docs = splitter.run(cleaner.run(raw_docs)["documents"])

len(splitted_docs["documents"])

"""
### FastEmbed Document Embedder

Here we are initializing the FastEmbed Document Embedder and using it to generate embeddings for the documents.
We are using a small and good model, `BAAI/bge-small-en-v1.5` and specifying the `parallel` parameter to 0 to use all available CPU cores for embedding generation.

⚠️ If you are running this notebook on Google Colab, please note that Google Colab only provides 2 CPU cores, so the embedding generation could be not as fast as it can be on a standard machine.

For more information on FastEmbed-Haystack integration, please refer to the [documentation](https://docs.haystack.deepset.ai/docs/fastembeddocumentembedder) and [API reference](https://docs.haystack.deepset.ai/reference/fastembed-embedders).
"""
logger.info("### FastEmbed Document Embedder")

document_embedder = FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5", parallel = 0, meta_fields_to_embed=["title"])
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(splitted_docs["documents"])

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

"""
## RAG Pipeline using Qwen 2.5 7B
"""
logger.info("## RAG Pipeline using Qwen 2.5 7B")



# from getpass import getpass

# os.environ["HF_API_TOKEN"] = getpass("Enter your Hugging Face Token: https://huggingface.co/settings/tokens ")

generator = HuggingFaceAPIChatGenerator(api_type="serverless_inference_api",
                              api_params={"model": "Qwen/Qwen2.5-7B-Instruct",
                                          "provider": "together"},
                                    generation_kwargs={"max_tokens":500})

template = [ChatMessage.from_user("""
Using only the information contained in these documents return a brief answer (max 50 words).
If the answer cannot be inferred from the documents, respond \"I don't know\".
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
""")]

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", parallel = 0, prefix="query:"))
query_pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
query_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
query_pipeline.add_component("generator", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "generator")

"""
## Try the pipeline
"""
logger.info("## Try the pipeline")

question = "Who is Dave Grohl?"

results = query_pipeline.run(
    {   "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)

for d in results['generator']['replies']:
  plogger.debug(d.text)

logger.info("\n\n[DONE]", bright=True)