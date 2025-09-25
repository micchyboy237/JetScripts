from IPython.display import Image
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from jet.logger import logger
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# RAG pipelines with Haystack + Zephyr 7B Beta 🪁

*Notebook by [Stefano Fiorucci](https://www.linkedin.com/in/stefano-fiorucci/) and [Tuana Celik](https://www.linkedin.com/in/tuanacelik/)*

We are going to build a nice Retrieval Augmented Generation pipeline for Rock music, using the 🏗️ **Haystack LLM orchestration framework** and a good LLM: 💬 [Zephyr 7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) (fine-tuned version of Mistral 7B V.01 that focuses on helpfulness and outperforms many larger models on the MT-Bench and AlpacaEval benchmarks)

## Install dependencies
- `wikipedia` is needed to download data from Wikipedia
- `haystack-ai` is the Haystack package
- `sentence_transformers` is needed for embeddings
- `transformers` is needed to use open-source LLMs
- `accelerate` and `bitsandbytes` are required to use quantized versions of these models (with smaller memory footprint)
"""
logger.info("# RAG pipelines with Haystack + Zephyr 7B Beta 🪁")

# %%capture
# ! pip install wikipedia haystack-ai transformers accelerate bitsandbytes sentence_transformers


"""
## Load data from Wikipedia

We are going to download the Wikipedia pages related to some Rock bands, using the python library `wikipedia`.

These pages are converted into Haystack Documents
"""
logger.info("## Load data from Wikipedia")

favourite_bands="""Audioslave
Blink-182
Dire Straits
Evanescence
Green Day
Muse (band)
Nirvana (band)
Sum 41
The Cure
The Smiths""".split("\n")


raw_docs=[]

for title in favourite_bands:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
## The Indexing Pipeline
"""
logger.info("## The Indexing Pipeline")


"""
We will save our final Documents in an `InMemoryDocumentStore`, a simple database which lives in memory.
"""
logger.info("We will save our final Documents in an `InMemoryDocumentStore`, a simple database which lives in memory.")

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

"""
Our indexing Pipeline transform the original Documents and save them in the Document Store.

It consists of several components:

- `DocumentCleaner`: performs a basic cleaning of the Documents
- `DocumentSplitter`: chunks each Document into smaller pieces (more appropriate for semantic search and RAG)
- `SentenceTransformersDocumentEmbedder`:
  - represent each Document as a vector (capturing its meaning).
  - we choose a good but not too big model from [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  - Also the metadata `title` is embedded, because it contains relevant information (`metadata_fields_to_embed` parameter).
  - We use the GPU for this expensive operation (`device` parameter).
- `DocumentWriter` just saves the Documents in the Document Store
"""
logger.info("Our indexing Pipeline transform the original Documents and save them in the Document Store.")

indexing = Pipeline()
indexing.add_component("cleaner", DocumentCleaner())
indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=2))
indexing.add_component("doc_embedder", SentenceTransformersDocumentEmbedder(model="thenlper/gte-large",
                                                                            device=ComponentDevice.from_str("cuda:0"),
                                                                            meta_fields_to_embed=["title"]))
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

indexing.connect("cleaner", "splitter")
indexing.connect("splitter", "doc_embedder")
indexing.connect("doc_embedder", "writer")

"""
Let's draw the indexing pipeline
"""
logger.info("Let's draw the indexing pipeline")

indexing.draw("indexing.png")
Image(filename='indexing.png')

"""
We finally run the indexing pipeline
"""
logger.info("We finally run the indexing pipeline")

indexing.run({"cleaner":{"documents":raw_docs}})

"""
Let's inspect the total number of chunked Documents and examine a Document
"""
logger.info("Let's inspect the total number of chunked Documents and examine a Document")

len(document_store.filter_documents())

document_store.filter_documents()[0].meta

plogger.debug(document_store.filter_documents()[0])
logger.debug(len(document_store.filter_documents()[0].embedding)) # embedding size

"""
## The RAG Pipeline

### `HuggingFaceLocalGenerator` with `zephyr-7b-beta`

- To load and manage Open Source LLMs in Haystack, we can use the `HuggingFaceLocalGenerator`.

- The LLM we choose is [Zephyr 7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), a fine-tuned version of Mistral 7B V.01 that focuses on helpfulness and outperforms many larger models on the MT-Bench and AlpacaEval benchmarks; the model was fine-tuned by the Hugging Face team.

- Since we are using a free Colab instance (with limited resources), we load the model using **4-bit quantization** (passing the appropriate `huggingface_pipeline_kwargs` to our Generator).
For an introduction to Quantization in Hugging Face Transformers, you can read [this simple blog post](https://huggingface.co/blog/merve/quantization).
"""
logger.info("## The RAG Pipeline")


generator = HuggingFaceLocalGenerator("HuggingFaceH4/zephyr-7b-beta",
                                 huggingface_pipeline_kwargs={"device_map":"auto",
                                               "model_kwargs":{"load_in_4bit":True,
                                                "bnb_4bit_use_double_quant":True,
                                                "bnb_4bit_quant_type":"nf4",
                                                "bnb_4bit_compute_dtype":torch.bfloat16}},
                                 generation_kwargs={"max_new_tokens": 350})

"""
Let's warm up the component and try the model...
"""
logger.info("Let's warm up the component and try the model...")

generator.warm_up()

rich.logger.debug(generator.run("Please write a rhyme about Italy."))

"""
Ok, nice!

### `PromptBuilder`

 It's a component that renders a prompt from a template string using Jinja2 engine.

 Let's setup our prompt builder, with a format like the following (appropriate for Zephyr):

 `"<|system|>\nSYSTEM MESSAGE</s>\n<|user|>\nUSER MESSAGE</s>\n<|assistant|>\n"`
"""
logger.info("### `PromptBuilder`")


prompt_template = """<|system|>Using the information contained in the context, give a comprehensive answer to the question.
If the answer is contained in the context, also report the source URL.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
  {% for doc in documents %}
  {{ doc.content }} URL:{{ doc.meta['url'] }}
  {% endfor %};
  Question: {{query}}
  </s>
<|assistant|>
"""
prompt_builder = PromptBuilder(template=prompt_template)

"""
### Let's create the RAG pipeline
"""
logger.info("### Let's create the RAG pipeline")


"""
Our RAG Pipeline finds Documents relevant to the user query and pass them to the LLM to generate a grounded answer.

It consists of several components:

- `SentenceTransformersTextEmbedder`: represent the query as a vector (capturing its meaning).
- `InMemoryEmbeddingRetriever`: finds the (top 5) Documents that are most similar to the query vector
- `PromptBuilder`
- `HuggingFaceLocalGenerator`
"""
logger.info("Our RAG Pipeline finds Documents relevant to the user query and pass them to the LLM to generate a grounded answer.")

rag = Pipeline()
rag.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large",
                                                                    device=ComponentDevice.from_str("cuda:0"))
rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
rag.add_component("prompt_builder", prompt_builder)
rag.add_component("llm", generator)

rag.connect("text_embedder", "retriever")
rag.connect("retriever.documents", "prompt_builder.documents")
rag.connect("prompt_builder.prompt", "llm.prompt")

"""
Visualize our pipeline!
"""
logger.info("Visualize our pipeline!")

rag.draw("rag.png")
Image(filename='rag.png')

"""
We create an utility function that runs the RAG pipeline and nicely prints the answer.
"""
logger.info("We create an utility function that runs the RAG pipeline and nicely prints the answer.")

def get_generative_answer(query):

  results = rag.run({
      "text_embedder": {"text": query},
      "prompt_builder": {"query": query}
    }
  )

  answer = results["llm"]["replies"][0]
  rich.logger.debug(answer)

"""
Let's try our RAG pipeline...
"""
logger.info("Let's try our RAG pipeline...")

get_generative_answer("What is the style of the Cure?")

get_generative_answer("Is the earth flat?")

"""
More questions to try...
"""
logger.info("More questions to try...")

nice_questions_to_try="""What was the original name of Sum 41?
What was the title of Nirvana's breakthrough album released in 1991?
Green Day's "American Idiot" is a rock opera. What's the story it tells?
What is the most well-known album by Blink-182?
Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?
Evanescence's "Bring Me to Life" features a male vocalist. Who is he, and how does his voice complement Amy Lee's in the song?
Was Ozzy Osbourne part of Blink 182?
Dire Straits' "Sultans of Swing" is a classic rock track. How does Mark Knopfler's guitar work in this song stand out to you?
What is Sum 41's debut studio album called?
Which member of Muse is the lead vocalist and primary songwriter?
Who was the lead singer of Audioslave?
Who are the members of Green Day?
When was Nirvana's first studio album, "Bleach," released?
Were the Smiths an influential band?
What is the name of Evanescence's debut album?
Which band was Morrissey the lead singer of before he formed The Smiths?
What is the title of The Cure's most famous and successful album?
Dire Straits' hit song "Money for Nothing" features a guest vocal by a famous artist. Who is this artist?
Who played the song "Like a stone"?""".split('\n')

q=random.choice(nice_questions_to_try)
logger.debug(q)
get_generative_answer(q)

q=random.choice(nice_questions_to_try)
logger.debug(q)
get_generative_answer(q)

logger.info("\n\n[DONE]", bright=True)