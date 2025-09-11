from botocore.client import Config
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_aws import Bedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
import os
import shutil
import {ItemTable} from "@theme/FeatureTables";


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
---
sidebar_label: Bedrock (Knowledge Bases)
---

# Bedrock (Knowledge Bases) Retriever

This guide will help you get started with the AWS Knowledge Bases [retriever](/docs/concepts/retrievers).

[Knowledge Bases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/) is an Amazon Web Services (AWS) offering which lets you quickly build RAG applications by using your private data to customize FM response.

Implementing `RAG` requires organizations to perform several cumbersome steps to convert data into embeddings (vectors), store the embeddings in a specialized vector database, and build custom integrations into the database to search and retrieve text relevant to the user’s query. This can be time-consuming and inefficient.

With `Knowledge Bases for Amazon Bedrock`, simply point to the location of your data in `Amazon S3`, and `Knowledge Bases for Amazon Bedrock` takes care of the entire ingestion workflow into your vector database. If you do not have an existing vector database, Amazon Bedrock creates an Amazon OpenSearch Serverless vector store for you. For retrievals, use the Langchain - Amazon Bedrock integration via the Retrieve API to retrieve relevant results for a user query from knowledge bases.

### Integration details


<ItemTable category="document_retrievers" item="AmazonKnowledgeBasesRetriever" />

## Setup

Knowledge Bases can be configured through [AWS Console](https://aws.amazon.com/console/) or by using [AWS SDKs](https://aws.amazon.com/developer/tools/). We will need the `knowledge_base_id` to instantiate the retriever.

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# Bedrock (Knowledge Bases) Retriever")



"""
### Installation

This retriever lives in the `langchain-aws` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-aws

"""
## Instantiation

Now we can instantiate our retriever:
"""
logger.info("## Instantiation")


retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="PUIJP4EQUA",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

"""
## Usage
"""
logger.info("## Usage")

query = "What did the president say about Ketanji Brown?"

retriever.invoke(query)

"""
## Use within a chain
"""
logger.info("## Use within a chain")


model_kwargs_claude = {"temperature": 0, "top_k": 10, "max_tokens_to_sample": 3000}

llm = Bedrock(model_id="anthropic.claude-v2", model_kwargs=model_kwargs_claude)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

qa(query)

"""
## API reference

For detailed documentation of all `AmazonKnowledgeBasesRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/aws/retrievers/langchain_aws.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)