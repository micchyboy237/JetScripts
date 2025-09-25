from apify_haystack import ApifyDatasetFromActorCall
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
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
# RAG: Extract and use website content for question answering with Apify-Haystack integration

Author: Jiri Spilka ([Apify](https://apify.com/jiri.spilka))

In this tutorial, we'll use the [apify-haystack](https://github.com/apify/apify-haystack/tree/main) integration to call [Website Content Crawler](https://apify.com/apify/website-content-crawler) and crawl and scrape text content from the [Haystack website](https://haystack.deepset.ai). Then, we'll use the [OpenAIDocumentEmbedder](https://docs.haystack.deepset.ai/docs/openaidocumentembedder) to compute text embeddings and the [InMemoryDocumentStore](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore) to store documents in a temporary in-memory database. The last step will be a retrieval augmented generation pipeline to answer users' questions from the scraped data.


## Install dependencies
"""
logger.info("# RAG: Extract and use website content for question answering with Apify-Haystack integration")

# !pip install apify-haystack haystack-ai

"""
## Set up the API keys

You need to have an Apify account and obtain [APIFY_API_TOKEN](https://docs.apify.com/platform/integrations/api).

# You also need an Ollama account and [OPENAI_API_KEY](https://platform.ollama.com/docs/quickstart)
"""
logger.info("## Set up the API keys")

# from getpass import getpass

# os.environ["APIFY_API_TOKEN"] = getpass("Enter YOUR APIFY_API_TOKEN")
# os.environ["OPENAI_API_KEY"] = getpass("Enter YOUR OPENAI_API_KEY")

"""
## Use the Website Content Crawler to scrape data from the haystack documentation

Now, let us call the Website Content Crawler using the Haystack component `ApifyDatasetFromActorCall`. First, we need to define parameters for the Website Content Crawler and then what data we need to save into the vector database.

The `actor_id` and detailed description of input parameters (variable `run_input`) can be found on the [Website Content Crawler input page](https://apify.com/apify/website-content-crawler/input-schema).

For this example, we will define `startUrls` and limit the number of crawled pages to five.
"""
logger.info("## Use the Website Content Crawler to scrape data from the haystack documentation")

actor_id = "apify/website-content-crawler"
run_input = {
    "maxCrawlPages": 5,  # limit the number of pages to crawl
    "startUrls": [{"url": "https://haystack.deepset.ai/"}],
}

"""
Next, we need to define a dataset mapping function. We need to know the output of the Website Content Crawler. Typically, it is a JSON object that looks like this (truncated for brevity):

```
[
  {
    "url": "https://haystack.deepset.ai/",
    "text": "Haystack | Haystack - Multimodal - AI - Architect a next generation AI app around all modalities, not just text ..."
  },
  {
    "url": "https://haystack.deepset.ai/tutorials/24_building_chat_app",
    "text": "Building a Conversational Chat App ... "
  },
]
```

We will convert this JSON to a Haystack `Document` using the `dataset_mapping_function` as follows:
"""
logger.info("Next, we need to define a dataset mapping function. We need to know the output of the Website Content Crawler. Typically, it is a JSON object that looks like this (truncated for brevity):")


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})

"""
And the definition of the `ApifyDatasetFromActorCall`:
"""
logger.info("And the definition of the `ApifyDatasetFromActorCall`:")


apify_dataset_loader = ApifyDatasetFromActorCall(
    actor_id=actor_id,
    run_input=run_input,
    dataset_mapping_function=dataset_mapping_function,
)

"""
Before actually running the Website Content Crawler, we need to define embedding function and document store:
"""
logger.info("Before actually running the Website Content Crawler, we need to define embedding function and document store:")


document_store = InMemoryDocumentStore()
docs_embedder = OpenAIDocumentEmbedder()

"""
After that, we can call the Website Content Crawler and print the scraped data:
"""
logger.info("After that, we can call the Website Content Crawler and print the scraped data:")

docs = apify_dataset_loader.run()
logger.debug(docs)

"""
Compute the embeddings and store them in the database:
"""
logger.info("Compute the embeddings and store them in the database:")

embeddings = docs_embedder.run(docs.get("documents"))
document_store.write_documents(embeddings["documents"])

"""
## Retrieval and LLM generative pipeline

Once we have the crawled data in the database, we can set up the classical retrieval augmented pipeline. Refer to the [RAG Haystack tutorial](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline) for details.
"""
logger.info("## Retrieval and LLM generative pipeline")


text_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store)
generator = OpenAIGenerator(model="llama3.2")

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

logger.debug("Initializing pipeline...")
pipe = Pipeline()
pipe.add_component("embedder", text_embedder)
pipe.add_component("retriever", retriever)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)

pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

"""
Now, you can ask questions about Haystack and get correct answers:
"""
logger.info("Now, you can ask questions about Haystack and get correct answers:")

question = "What is haystack?"

response = pipe.run({"embedder": {"text": question}, "prompt_builder": {"question": question}})

logger.debug(f"question: {question}")
logger.debug(f"answer: {response['llm']['replies'][0]}")

logger.info("\n\n[DONE]", bright=True)