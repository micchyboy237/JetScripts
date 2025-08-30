from IPython.display import Image
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice
from jet.logger import CustomLogger
from pprint import pprint
import os
import random
import rich
import shutil
import torch
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


"""
# 🏆🎬 RAG with Llama 3.1 and Haystack

  <img src="https://img-cdn.inc.com/image/upload/w_1280,ar_16:9,c_fill,g_auto,q_auto:best/images/panoramic/meta-llama3-inc_539927_dhgoal.webp" width="380"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://haystack.deepset.ai/images/haystack-ogimage.png" width="430" style="display:inline;">



Simple RAG example on the Oscars using [Llama 3.1 open models](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) and the [Haystack LLM framework](https://haystack.deepset.ai/).

## Installation
"""
logger.info("# 🏆🎬 RAG with Llama 3.1 and Haystack")

# ! pip install haystack-ai "transformers>=4.43.1" sentence-transformers accelerate bitsandbytes

"""
## Authorization

- you need an Hugging Face account
- you need to accept Meta conditions here: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct and wait for the authorization
"""
logger.info("## Authorization")

# import getpass, os


# os.environ["HF_API_TOKEN"] = getpass.getpass("Your Hugging Face token")

"""
## RAG with Llama-3.1-8B-Instruct (about the Oscars) 🏆🎬
"""
logger.info("## RAG with Llama-3.1-8B-Instruct (about the Oscars) 🏆🎬")

# ! pip install wikipedia

"""
### Load data from Wikipedia
"""
logger.info("### Load data from Wikipedia")



title = "96th_Academy_Awards"
page = wikipedia.page(title=title, auto_suggest=False)
raw_docs = [Document(content=page.content, meta={"title": page.title, "url":page.url})]

"""
### Indexing Pipeline
"""
logger.info("### Indexing Pipeline")


document_store = InMemoryDocumentStore()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))

indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        model="Snowflake/snowflake-arctic-embed-l",  # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
        device=ComponentDevice.from_str("cuda:0"),    # load the model on GPU
    ))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"splitter":{"documents":raw_docs}})

"""
### RAG Pipeline
"""
logger.info("### RAG Pipeline")


template = [ChatMessage.from_user("""
Using the information contained in the context, give a comprehensive answer to the question.
If the answer cannot be deduced from the context, do not give an answer.

Context:
  {% for doc in documents %}
  {{ doc.content }} URL:{{ doc.meta['url'] }}
  {% endfor %};
  Question: {{query}}

""")]
prompt_builder = ChatPromptBuilder(template=template)

"""
Here, we use the [`HuggingFaceLocalChatGenerator`](https://docs.haystack.deepset.ai/docs/huggingfacelocalchatgenerator), loading the model in Colab with 4-bit quantization.
"""
logger.info("Here, we use the [`HuggingFaceLocalChatGenerator`](https://docs.haystack.deepset.ai/docs/huggingfacelocalchatgenerator), loading the model in Colab with 4-bit quantization.")


generator = HuggingFaceLocalChatGenerator(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    huggingface_pipeline_kwargs={"device_map":"auto",
                                  "model_kwargs":{"load_in_4bit":True,
                                                  "bnb_4bit_use_double_quant":True,
                                                  "bnb_4bit_quant_type":"nf4",
                                                  "bnb_4bit_compute_dtype":torch.bfloat16}},
    generation_kwargs={"max_new_tokens": 500})

generator.warm_up()


query_pipeline = Pipeline()

query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(
        model="Snowflake/snowflake-arctic-embed-l",  # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
        device=ComponentDevice.from_str("cuda:0"),  # load the model on GPU
        prefix="Represent this sentence for searching relevant passages: ",  # as explained in the model card (https://huggingface.co/Snowflake/snowflake-arctic-embed-l#using-huggingface-transformers), queries should be prefixed
    ))
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
query_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
query_pipeline.add_component("generator", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "generator")

"""
### Let's ask some questions!
"""
logger.info("### Let's ask some questions!")

def get_generative_answer(query):

  results = query_pipeline.run({
      "text_embedder": {"text": query},
      "prompt_builder": {"query": query}
    }
  )

  answer = results["generator"]["replies"][0].text
  rich.logger.debug(answer)

get_generative_answer("Who won the Best Picture Award in 2024?")

get_generative_answer("What was the box office performance of the Best Picture nominees?")

get_generative_answer("What was the reception of the ceremony")

get_generative_answer("Can you name some of the films that got multiple nominations?")

get_generative_answer("Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?")

"""
---
This is a simple demo.
We can improve the RAG Pipeline in several ways, including better preprocessing the input.

To use Llama 3 models in Haystack, you also have **other options**:
- [LlamaCppGenerator](https://docs.haystack.deepset.ai/docs/llamacppgenerator) and [OllamaGenerator](https://docs.haystack.deepset.ai/docs/ollamagenerator): using the GGUF quantized format, these solutions are ideal to run LLMs on standard machines (even without GPUs).
- [HuggingFaceAPIChatGenerator](https://docs.haystack.deepset.ai/docs/huggingfaceapichatgenerator), which allows you to query a the Hugging Face API, a local TGI container or a (paid) HF Inference Endpoint. TGI is a toolkit for efficiently deploying and serving LLMs in production.
- [vLLM via OllamaFunctionCallingAdapterChatGenerator](https://haystack.deepset.ai/integrations/vllm): high-throughput and memory-efficient inference and serving engine for LLMs.

(*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
"""
logger.info("This is a simple demo.")

logger.info("\n\n[DONE]", bright=True)