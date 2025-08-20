from jet.logger import CustomLogger
from llama_index.embeddings.ibm import WatsonxEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/ibm_watsonx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# IBM watsonx.ai

>WatsonxEmbeddings is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) embedding models.

This example shows how to communicate with `watsonx.ai` embedding models using the `LlamaIndex` Embeddings API.

## Setting up

Install the `llama-index-embeddings-ibm` package:
"""
logger.info("# IBM watsonx.ai")

# !pip install -qU llama-index-embeddings-ibm

"""
The cell below defines the credentials required to work with watsonx Embeddings.

**Action:** Provide the IBM Cloud user API key. For details, see
[Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
"""
logger.info("The cell below defines the credentials required to work with watsonx Embeddings.")

# from getpass import getpass

# watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key

"""
Additionally, you can pass additional secrets as an environment variable:
"""
logger.info("Additionally, you can pass additional secrets as an environment variable:")


os.environ["WATSONX_URL"] = "your service instance url"
os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
os.environ[
    "WATSONX_INSTANCE_ID"
] = "your instance_id for accessing the CPD cluster"

"""
## Load the model

You might need to adjust embedding parameters for different tasks:
"""
logger.info("## Load the model")

truncate_input_tokens = 3

"""
Initialize the `WatsonxEmbeddings` class with the previously set parameter.


**Note**: 

- To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
- Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas URL.


You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported foundation models](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes).
"""
logger.info("Initialize the `WatsonxEmbeddings` class with the previously set parameter.")


watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    truncate_input_tokens=truncate_input_tokens,
)

"""
Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).
"""
logger.info("Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).")

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    truncate_input_tokens=truncate_input_tokens,
)

"""
## Usage

### Embed query
"""
logger.info("## Usage")

query = "Example query."

query_result = watsonx_embedding.get_query_embedding(query)
logger.debug(query_result[:5])

"""
### Embed list of texts
"""
logger.info("### Embed list of texts")

texts = ["This is a content of one document", "This is another document"]

doc_result = watsonx_embedding.get_text_embedding_batch(texts)
logger.debug(doc_result[0][:5])

logger.info("\n\n[DONE]", bright=True)