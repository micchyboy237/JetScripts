from jet.logger import logger
from langchain_db2 import db2vs
from langchain_db2.db2vs import DB2VS
from langchain_ibm import ChatWatsonx
from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import WatsonxLLM
from langchain_ibm import WatsonxRerank
from langchain_ibm import WatsonxToolkit
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
# IBM

LangChain integrations related to IBM technologies, including the
[IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) platform and DB2 database.

## Watsonx AI
IBM® watsonx.ai™ AI studio is part of the IBM [watsonx](https://www.ibm.com/watsonx)™ AI and data platform, bringing together new generative
AI capabilities powered by [foundation models](https://www.ibm.com/products/watsonx-ai/foundation-models) and traditional machine learning (ML)
into a powerful studio spanning the AI lifecycle. Tune and guide models with your enterprise data to meet your needs with easy-to-use tools for
building and refining performant prompts. With watsonx.ai, you can build AI applications in a fraction of the time and with a fraction of the data.
Watsonx.ai offers:

- **Multi-model variety and flexibility:** Choose from IBM-developed, open-source and third-party models, or build your own model.
- **Differentiated client protection:** IBM stands behind IBM-developed models and indemnifies the client against third-party IP claims.
- **End-to-end AI governance:** Enterprises can scale and accelerate the impact of AI with trusted data across the business, using data wherever it resides.
- **Hybrid, multi-cloud deployments:** IBM provides the flexibility to integrate and deploy your AI workloads into your hybrid-cloud stack of choice.


### Installation and Setup

Install the integration package with
"""
logger.info("# IBM")

pip install -qU langchain-ibm

"""
Get an IBM watsonx.ai api key and set it as an environment variable (`WATSONX_APIKEY`)
"""
logger.info("Get an IBM watsonx.ai api key and set it as an environment variable (`WATSONX_APIKEY`)")


os.environ["WATSONX_APIKEY"] = "your IBM watsonx.ai api key"

"""
### Chat Model

#### ChatWatsonx

See a [usage example](/docs/integrations/chat/ibm_watsonx).
"""
logger.info("### Chat Model")


"""
### LLMs

#### WatsonxLLM

See a [usage example](/docs/integrations/llms/ibm_watsonx).
"""
logger.info("### LLMs")


"""
### Embedding Models

#### WatsonxEmbeddings

See a [usage example](/docs/integrations/text_embedding/ibm_watsonx).
"""
logger.info("### Embedding Models")


"""
### Reranker

#### WatsonxRerank

See a [usage example](/docs/integrations/retrievers/ibm_watsonx_ranker).
"""
logger.info("### Reranker")


"""
### Toolkit

#### WatsonxToolkit

See a [usage example](/docs/integrations/tools/ibm_watsonx).
"""
logger.info("### Toolkit")


"""
## DB2

### Vector stores

#### IBM DB2 Vector Store and Vector Search

The IBM DB2 relational database v12.1.2 and above offers the abilities of vector store
and vector search. Installation of `langchain-db2` package will give Langchain users
the support of DB2 vector store and vector search.

See detailed usage examples in the guide [here](/docs/integrations/vectorstores/db2).

Installation: This is a separate package for vector store feature only and can be run
without the `langchain-ibm` package.
"""
logger.info("## DB2")

pip install -U langchain-db2

"""
Usage:
"""
logger.info("Usage:")


logger.info("\n\n[DONE]", bright=True)