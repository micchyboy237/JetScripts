from IPython.display import Image
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from jet.logger import logger
from pprint import pprint
import os
import random
import rich
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
# Build with Gemma and Haystack

<img src="https://huggingface.co/blog/assets/gemma/Gemma-logo-small.png" width="200" style="display:inline;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://haystack.deepset.ai/images/haystack-ogimage.png" width="430" style="display:inline;">



We will see what we can build with the new [Google Gemma open models](https://blog.google/technology/developers/gemma-open-models/) and the [Haystack LLM framework](https://haystack.deepset.ai/).

## Installation
"""
logger.info("# Build with Gemma and Haystack")

# ! pip install haystack-ai "huggingface_hub>=0.22.0"

"""
## Authorization

- you need an Hugging Face account
- you need to accept Google conditions here: https://huggingface.co/google/gemma-7b-it and wait for the authorization
"""
logger.info("## Authorization")

# import getpass, os


# os.environ["HF_API_TOKEN"] = getpass.getpass("Your Hugging Face token")

"""
## Chat with Gemma (travel assistant) 🛩

For simplicity, we call the model using the free Hugging Face Inference API with the `HuggingFaceAPIChatGenerator`.

(We might also load it in Colab using the `HuggingFaceLocalChatGenerator` in a quantized version).
"""
logger.info("## Chat with Gemma (travel assistant) 🛩")


generator = HuggingFaceAPIChatGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "google/gemma-7b-it"},
    generation_kwargs={"max_new_tokens": 350})

messages = []

while True:
  msg = input("Enter your message or Q to exit\n🧑 ")
  if msg=="Q":
    break
  messages.append(ChatMessage.from_user(msg))
  response = generator.run(messages=messages)
  assistant_resp = response['replies'][0]
  logger.debug("🤖 "+assistant_resp.text)
  messages.append(assistant_resp)

"""
## RAG with Gemma (about Rock music) 🎸
"""
logger.info("## RAG with Gemma (about Rock music) 🎸")

# ! pip install wikipedia

"""
### Load data from Wikipedia
"""
logger.info("### Load data from Wikipedia")

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
### Indexing Pipeline
"""
logger.info("### Indexing Pipeline")


document_store = InMemoryDocumentStore()

indexing = Pipeline()
indexing.add_component("cleaner", DocumentCleaner())
indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=2))
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

indexing.connect("cleaner", "splitter")
indexing.connect("splitter", "writer")

indexing.run({"cleaner":{"documents":raw_docs}})

document_store.filter_documents()[0].meta

"""
### RAG Pipeline
"""
logger.info("### RAG Pipeline")


prompt_template = """
<start_of_turn>user
Using the information contained in the context, give a comprehensive answer to the question.
If the answer is contained in the context, also report the source URL.
If the answer cannot be deduced from the context, do not give an answer.

Context:
  {% for doc in documents %}
  {{ doc.content }} URL:{{ doc.meta['url'] }}
  {% endfor %};
  Question: {{query}}<end_of_turn>

<start_of_turn>model
"""
prompt_builder = PromptBuilder(template=prompt_template)

"""
Here, we use the `HuggingFaceAPIGenerator` since it is not a chat setting and we don't envision multi-turn conversations but just RAG.
"""
logger.info("Here, we use the `HuggingFaceAPIGenerator` since it is not a chat setting and we don't envision multi-turn conversations but just RAG.")


generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "google/gemma-7b-it"},
    generation_kwargs={"max_new_tokens": 500})


rag = Pipeline()
rag.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=5))
rag.add_component("prompt_builder", prompt_builder)
rag.add_component("llm", generator)

rag.connect("retriever.documents", "prompt_builder.documents")
rag.connect("prompt_builder.prompt", "llm.prompt")

"""
### Let's ask some questions!
"""
logger.info("### Let's ask some questions!")

def get_generative_answer(query):

  results = rag.run({
      "retriever": {"query": query},
      "prompt_builder": {"query": query}
    }
  )

  answer = results["llm"]["replies"][0]
  rich.logger.debug(answer)

get_generative_answer("Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?")

nice_questions_to_try="""What was the original name of Sum 41?
What was the title of Nirvana's breakthrough album released in 1991?
Green Day's "American Idiot" is a rock opera. What's the story it tells?
Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?
Evanescence's "Bring Me to Life" features a male vocalist. Who is he, and how does his voice complement Amy Lee's in the song?
What is Sum 41's debut studio album called?
Who was the lead singer of Audioslave?
When was Nirvana's first studio album, "Bleach," released?
Were the Smiths an influential band?
What is the name of Evanescence's debut album?
Which band was Morrissey the lead singer of before he formed The Smiths?
Dire Straits' hit song "Money for Nothing" features a guest vocal by a famous artist. Who is this artist?
Who played the song "Like a stone"?""".split('\n')

q=random.choice(nice_questions_to_try)
logger.debug(q)
get_generative_answer(q)

"""
This is a simple demo.
We can improve the RAG Pipeline using better retrieval techniques: Embedding Retrieval, Hybrid Retrieval...

(*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
"""
logger.info("This is a simple demo.")

logger.info("\n\n[DONE]", bright=True)