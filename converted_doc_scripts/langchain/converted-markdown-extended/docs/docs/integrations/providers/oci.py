from jet.logger import logger
from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.chat_models import ChatOCIModelDeployment
from langchain_oci.embeddings import OCIGenAIEmbeddings
from langchain_oci.llms import OCIGenAI
from langchain_oci.llms import OCIModelDeploymentLLM
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
# Oracle Cloud Infrastructure (OCI)

The `LangChain` integrations related to [Oracle Cloud Infrastructure](https://www.oracle.com/artificial-intelligence/).

## OCI Generative AI
> Oracle Cloud Infrastructure (OCI) [Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) is a fully managed service that provides a set of state-of-the-art,
> customizable large language models (LLMs) that cover a wide range of use cases, and which are available through a single API.
> Using the OCI Generative AI service you can access ready-to-use pretrained models, or create and host your own fine-tuned
> custom models based on your own data on dedicated AI clusters.

To use, you should have the latest `oci` python SDK and the langchain_community package installed.
"""
logger.info("# Oracle Cloud Infrastructure (OCI)")

pip install -U langchain_oci

"""
See [chat](/docs/integrations/llms/oci_generative_ai), [complete](/docs/integrations/chat/oci_generative_ai), and [embedding](/docs/integrations/text_embedding/oci_generative_ai) usage examples.
"""
logger.info("See [chat](/docs/integrations/llms/oci_generative_ai), [complete](/docs/integrations/chat/oci_generative_ai), and [embedding](/docs/integrations/text_embedding/oci_generative_ai) usage examples.")




"""
## OCI Data Science Model Deployment Endpoint

> [OCI Data Science](https://docs.oracle.com/en-us/iaas/data-science/using/home.htm) is a
> fully managed and serverless platform for data science teams. Using the OCI Data Science
> platform you can build, train, and manage machine learning models, and then deploy them
> as an OCI Model Deployment Endpoint using the
> [OCI Data Science Model Deployment Service](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm).

To use, you should have the latest `oracle-ads` python SDK installed.
"""
logger.info("## OCI Data Science Model Deployment Endpoint")

pip install -U oracle-ads

"""
See [chat](/docs/integrations/chat/oci_data_science) and [complete](/docs/integrations/llms/oci_model_deployment_endpoint) usage examples.
"""
logger.info("See [chat](/docs/integrations/chat/oci_data_science) and [complete](/docs/integrations/llms/oci_model_deployment_endpoint) usage examples.")



logger.info("\n\n[DONE]", bright=True)