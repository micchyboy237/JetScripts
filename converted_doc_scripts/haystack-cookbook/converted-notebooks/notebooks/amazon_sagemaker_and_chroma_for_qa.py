from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from jet.logger import CustomLogger
from pathlib import Path
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
# Question Answering with Amazon Sagemaker, Chroma and Haystack

*Notebook by [Sara Zanzottera](https://www.zansara.dev/) and [Bilge Yucel](https://www.linkedin.com/in/bilge-yucel/)*

[Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/) is a comprehensive, fully managed machine learning service
that allows data scientists and developers to build, train, and deploy ML models efficiently. You can choose from various foundation models to find the one best suited for your use case.

In this notebook, we'll go through the process of **creating a generative question answering application** using the newly added [Amazon Sagemaker integration](https://haystack.deepset.ai/integrations/amazon-sagemaker) with [Haystack](https://github.com/deepset-ai/haystack) and [Chroma](https://haystack.deepset.ai/integrations/chroma-documentstore) to store our documents efficiently. The demo will illustrate the step-by-step development of a QA application using some Wikipedia pages about NASA's Mars missions 🚀

## Setup the Development Environment

### Install dependencies
"""
logger.info("# Question Answering with Amazon Sagemaker, Chroma and Haystack")

# %%bash

pip install chroma-haystack amazon-sagemaker-haystack wikipedia typing_extensions

"""
## Deploy a model on Sagemaker

To use Amazon Sagemaker's models, you first need to deploy them. In this example we'll be using Falcon 7B Instruct BF16, so make sure to deploy such model on your account before proceeding.

For help you can check out:
- Amazon Sagemaker Jumpstart [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html).
- [This notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text-generation-falcon.ipynb) on how to deploy Falcon models programmatically with a notebook
- [This blogpost](https://aws.amazon.com/blogs/machine-learning/build-production-ready-generative-ai-applications-for-enterprise-search-using-haystack-pipelines-and-amazon-sagemaker-jumpstart-with-llms/) about deploying models on Sagemaker for Haystack 1.x

### API Keys

To use Amazon Sagemaker, you need to set a few environment variables: `AWS ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and often to indicate the region by setting `AWS_REGION`. Once logged into your account, locate these keys under the IAM user's "Security Credentials" section. For detailed guidance, refer to the documentation on [Managing access keys for IAM users](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).
"""
logger.info("## Deploy a model on Sagemaker")

# from getpass import getpass

# os.environ["AWS_ACCESS_KEY_ID"] = getpass("aws_access_key_id: ")
# os.environ["AWS_SECRET_ACCESS_KEY"] = getpass("aws_secret_access_key: ")
os.environ["AWS_REGION"] = input("aws_region_name: ")

"""
## Load data from Wikipedia

We are going to download the Wikipedia pages related to NASA's martian rovers using the python library `wikipedia`.

These pages are converted into Haystack Documents.
"""
logger.info("## Load data from Wikipedia")


wiki_pages = [
    "Ingenuity_(helicopter)",
    "Perseverance_(rover)",
    "Curiosity_(rover)",
    "Opportunity_(rover)",
    "Spirit_(rover)",
    "Sojourner_(rover)"
]

raw_docs=[]
for title in wiki_pages:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
## Building the Indexing Pipeline

Our indexing pipeline will preprocess the provided Wikipedia pages by cleaning and splitting it into chunks before storing them in [ChromaDocumentStore](https://docs.haystack.deepset.ai/v2.0/docs/chroma-document-store).

Let’s run the pipeline below and index our file to our document store:
"""
logger.info("## Building the Indexing Pipeline")



document_store = ChromaDocumentStore()

cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "writer")

"""
Run the pipeline with the files you want to index (note that this step may take some time):
"""
logger.info("Run the pipeline with the files you want to index (note that this step may take some time):")

indexing_pipeline.run({"cleaner":{"documents":raw_docs}})

"""
## Building the Query Pipeline

Let’s create another pipeline to query our application. In this pipeline, we’ll use [ChromaQueryTextRetriever](https://docs.haystack.deepset.ai/v2.0/docs/chromaqueryretriever) to retrieve relevant information from the ChromaDocumentStore and a Falcon 7B Instruct BF16 model to generate answers with [SagemakerGenerator](https://docs.haystack.deepset.ai/v2.0/docs/sagemakergenerator).

Next, we'll create a prompt for our task using the Retrieval-Augmented Generation (RAG) approach with [PromptBuilder](https://docs.haystack.deepset.ai/v2.0/docs/promptbuilder). This prompt will help generate answers by considering the provided context. Finally, we'll connect these three components to complete the pipeline.
"""
logger.info("## Building the Query Pipeline")


retriever = ChromaQueryTextRetriever(document_store=document_store, top_k=3)

model = 'jumpstart-dft-hf-llm-falcon-7b-instruct-bf16'
generator = SagemakerGenerator(model=model, generation_kwargs={"max_new_tokens":256})
template = """
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Answer based on the information above: {{question}}
"""
prompt_builder = PromptBuilder(template=template)

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

"""
Ask your question and learn about the Amazon Sagemaker service using Amazon Sagemaker models!
"""
logger.info("Ask your question and learn about the Amazon Sagemaker service using Amazon Sagemaker models!")

question = "When did Opportunity land?"
response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

logger.debug(response["llm"]["replies"][0])

question = "Is Ingenuity mission over?"
response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

logger.debug(response["llm"]["replies"][0])

question = "What was the name of the first NASA rover to land on Mars?"
response = rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

logger.debug(response["llm"]["replies"][0])

logger.info("\n\n[DONE]", bright=True)