from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.postprocessor.ibm import WatsonxRerank
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/ibm_watsonx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# IBM watsonx.ai

>WatsonxRerank is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) Rerank.

The aim of these examples is to show how to take advantage of `watsonx.ai` Rerank, Embeddings and LLMs using the `LlamaIndex` postprocessor API.

## Setting up

Install required packages:
"""
logger.info("# IBM watsonx.ai")

# %pip install -qU llama-index
# %pip install -qU llama-index-llms-ibm
# %pip install -qU llama-index-postprocessor-ibm
# %pip install -qU llama-index-embeddings-ibm

"""
The cell below defines the credentials required to work with watsonx Foundation Models, Embeddings and Rerank.

**Action:** Provide the IBM Cloud user API key. For details, see
[Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
"""
logger.info("The cell below defines the credentials required to work with watsonx Foundation Models, Embeddings and Rerank.")

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
**Note**: 

- To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
- Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas URL.

Provide `PROJECT_ID` that will be used for initialize each watsonx integration instance.
"""
logger.info("In this example, we’ll use the `project_id` and Dallas URL.")

PROJECT_ID = "PASTE YOUR PROJECT_ID HERE"
URL = "https://us-south.ml.cloud.ibm.com"

"""
## Download data and load documents
"""
logger.info("## Download data and load documents")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Load the Rerank

You might need to adjust rerank parameters for different tasks:
"""
logger.info("## Load the Rerank")

truncate_input_tokens = 512

"""
#### Initialize `WatsonxRerank` instance.

You need to specify the `model_id` that will be used for rerank. You can find the list of all the available models in [Supported reranker models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#rerank).
"""
logger.info("#### Initialize `WatsonxRerank` instance.")


watsonx_rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    top_n=2,
    url=URL,
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)

"""
Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).
"""
logger.info("Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).")


watsonx_rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    url=URL,
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="5.1",
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)

"""
## Load the embedding model

#### Initialize the `WatsonxEmbeddings` instance.

>For more information about `WatsonxEmbeddings` please refer to the sample notebook: <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/ibm_watsonx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

You might need to adjust embedding parameters for different tasks:
"""
logger.info("## Load the embedding model")

truncate_input_tokens = 512

"""
You need to specify the `model_id` that will be used for embedding. You can find the list of all the available models in [Supported embedding models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#embed).
"""
logger.info("You need to specify the `model_id` that will be used for embedding. You can find the list of all the available models in [Supported embedding models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#embed).")


watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=URL,
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)

"""
Change default settings
"""
logger.info("Change default settings")


Settings.chunk_size = 512

"""
#### Build index
"""
logger.info("#### Build index")


index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=watsonx_embedding
)

"""
## Load the LLM

#### Initialize the `WatsonxLLM` instance.

>For more information about `WatsonxLLM` please refer to the sample notebook: <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ibm_watsonx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported foundation models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx).

You might need to adjust model `parameters` for different models or tasks. For details, refer to [Available MetaNames](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).
"""
logger.info("## Load the LLM")

max_new_tokens = 128


watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=URL,
    project_id=PROJECT_ID,
    max_new_tokens=max_new_tokens,
)

"""
## Send a query

#### Retrieve top 10 most relevant nodes, then filter with `WatsonxRerank`
"""
logger.info("## Send a query")

query_engine = index.as_query_engine(
    llm=watsonx_llm,
    similarity_top_k=10,
    node_postprocessors=[watsonx_rerank],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)


pprint_response(response, show_source=True)

"""
#### Directly retrieve top 2 most similar nodes
"""
logger.info("#### Directly retrieve top 2 most similar nodes")

query_engine = index.as_query_engine(
    llm=watsonx_llm,
    similarity_top_k=2,
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

"""
Retrieved context is irrelevant and response is hallucinated.
"""
logger.info("Retrieved context is irrelevant and response is hallucinated.")

pprint_response(response, show_source=True)

logger.info("\n\n[DONE]", bright=True)