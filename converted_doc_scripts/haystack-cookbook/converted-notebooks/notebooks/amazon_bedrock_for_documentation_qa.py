from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from jet.logger import logger
from pathlib import Path
import os
import shutil
import time


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
# PDF-Based Question Answering with Amazon Bedrock and Haystack

*Notebook by [Bilge Yucel](https://www.linkedin.com/in/bilge-yucel/)*

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that provides high-performing foundation models from leading AI startups and Amazon through a single API. You can choose from various foundation models to find the one best suited for your use case.

In this notebook, we'll go through the process of **creating a generative question answering application** tailored for PDF files using the newly added [Amazon Bedrock integration](https://haystack.deepset.ai/integrations/amazon-bedrock) with [Haystack](https://github.com/deepset-ai/haystack) and [OpenSearch](https://haystack.deepset.ai/integrations/opensearch-document-store) to store our documents efficiently. The demo will illustrate the step-by-step development of a QA application designed specifically for the Bedrock documentation, demonstrating the power of Bedrock in the process 🚀

## Setup the Development Environment

### Install dependencies
"""
logger.info("# PDF-Based Question Answering with Amazon Bedrock and Haystack")

# %%bash

pip install -q opensearch-haystack amazon-bedrock-haystack pypdf

"""
### Download Files

For this application, we'll use the user guide of Amazon Bedrock. Amazon Bedrock provides the [PDF form of its guide](https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf). Let's download it!
"""
logger.info("### Download Files")

# !wget "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf"

"""
> Note: You can code to download the PDF to `/content/bedrock-documentation.pdf` directory as an alternative👇🏼
"""



"""
### Initialize an OpenSearch Instance on Colab

[OpenSearch](https://opensearch.org/) is a fully open source search and analytics engine and is compatible with the [Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html) that’s helpful if you’d like to deploy, operate, and scale your OpenSearch cluster later on.

Let’s install OpenSearch and start an instance on Colab. For other installation options, check out [OpenSearch documentation](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/index/).
"""
logger.info("### Initialize an OpenSearch Instance on Colab")

# !wget https://artifacts.opensearch.org/releases/bundle/opensearch/2.11.1/opensearch-2.11.1-linux-x64.tar.gz
# !tar -xvf opensearch-2.11.1-linux-x64.tar.gz
# !chown -R daemon:daemon opensearch-2.11.1
# !sudo echo 'plugins.security.disabled: true' >> opensearch-2.11.1/config/opensearch.yml

# %%bash --bg
cd opensearch-2.11.1 && sudo -u daemon -- ./bin/opensearch

"""
> OpenSearch needs 30 seconds for a fully started server
"""


time.sleep(30)

"""
### API Keys

To use Amazon Bedrock, you need `aws_access_key_id`, `aws_secret_access_key`, and indicate the `aws_region_name`. Once logged into your account, locate these keys under the IAM user's "Security Credentials" section. For detailed guidance, refer to the documentation on [Managing access keys for IAM users](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).
"""
logger.info("### API Keys")

# from getpass import getpass

# os.environ["AWS_ACCESS_KEY_ID"] = getpass("aws_access_key_id: ")
# os.environ["AWS_SECRET_ACCESS_KEY"] = getpass("aws_secret_access_key: ")
os.environ["AWS_DEFAULT_REGION"] = input("aws_region_name: ")

"""
## Building the Indexing Pipeline

Our indexing pipeline will convert the PDF file into a Haystack Document using [PyPDFToDocument](https://docs.haystack.deepset.ai/v2.0/docs/pypdftodocument) and preprocess it by cleaning and splitting it into chunks before storing them in [OpenSearchDocumentStore](https://docs.haystack.deepset.ai/v2.0/docs/opensearch-document-store).

Let’s run the pipeline below and index our file to our document store:
"""
logger.info("## Building the Indexing Pipeline")



document_store = OpenSearchDocumentStore()

converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "writer")

"""
Run the pipeline with the pdf. This might take ~4mins if you're running this notebook on CPU.
"""
logger.info("Run the pipeline with the pdf. This might take ~4mins if you're running this notebook on CPU.")

indexing_pipeline.run({"converter": {"sources": [Path("/content/bedrock-ug.pdf")]}})

"""
## Building the Query Pipeline

Let’s create another pipeline to query our application. In this pipeline, we’ll use [OpenSearchBM25Retriever](https://docs.haystack.deepset.ai/docs/opensearchbm25retriever) to retrieve relevant information from the OpenSearchDocumentStore and an Amazon Nova model `amazon.nova-pro-v1:0` to generate answers with [AmazonChatBedrockGenerator](https://docs.haystack.deepset.ai/docs/amazonbedrockchatgenerator). You can select and test different models using the dropdown on right.

Next, we'll create a prompt for our task using the Retrieval-Augmented Generation (RAG) approach with [ChatPromptBuilder](https://docs.haystack.deepset.ai/docs/chatpromptbuilder). This prompt will help generate answers by considering the provided context. Finally, we'll connect these three components to complete the pipeline.
"""
logger.info("## Building the Query Pipeline")


retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=15)

bedrock_model = 'amazon.nova-lite-v1:0'
generator = AmazonBedrockChatGenerator(model=bedrock_model)
template = """
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Please answer the question based on the given information from Amazon Bedrock documentation.

{{query}}
"""
prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(template)], required_variables="*")

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

"""
Ask your question and learn about the Amazon Bedrock service using Amazon Bedrock models!
"""
logger.info("Ask your question and learn about the Amazon Bedrock service using Amazon Bedrock models!")

question = "What is Amazon Bedrock?"
response = rag_pipeline.run({"query": question})

logger.debug(response["llm"]["replies"][0].text)

"""
### Other Queries

You can also try these queries:

* How can I setup Amazon Bedrock?
* How can I finetune foundation models?
* How should I form my prompts for Amazon Titan models?
* How should I form my prompts for Claude models?
"""
logger.info("### Other Queries")

logger.info("\n\n[DONE]", bright=True)