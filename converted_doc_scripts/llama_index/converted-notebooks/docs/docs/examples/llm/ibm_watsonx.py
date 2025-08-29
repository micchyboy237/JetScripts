from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.ibm import WatsonxLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ibm_watsonx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# IBM watsonx.ai

>WatsonxLLM is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.

The aim of these examples is to show how to communicate with `watsonx.ai` models using the `LlamaIndex` LLMs API.

## Setting up

Install the `llama-index-llms-ibm` package:
"""
logger.info("# IBM watsonx.ai")

# !pip install -qU llama-index-llms-ibm

"""
The cell below defines the credentials required to work with watsonx Foundation Model inferencing.

**Action:** Provide the IBM Cloud user API key. For details, see
[Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
"""
logger.info("The cell below defines the credentials required to work with watsonx Foundation Model inferencing.")

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

You might need to adjust model `parameters` for different models or tasks. For details, refer to [Available MetaNames](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).
"""
logger.info("## Load the model")

temperature = 0.5
max_new_tokens = 50
additional_params = {
    "decoding_method": "sample",
    "min_new_tokens": 1,
    "top_k": 50,
    "top_p": 1,
}

"""
Initialize the `WatsonxLLM` class with the previously set parameters.


**Note**: 

- To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
- Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas URL.


You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported foundation models](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes).
"""
logger.info("Initialize the `WatsonxLLM` class with the previously set parameters.")


watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)

"""
Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).
"""
logger.info("Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).")

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)

"""
Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_working_with_class_and_prompt_tuner.html).
"""
logger.info("Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_working_with_class_and_prompt_tuner.html).")

watsonx_llm = WatsonxLLM(
    deployment_id="PASTE YOUR DEPLOYMENT_ID HERE",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)

"""
## Create a Completion
Call the model directly using a string type prompt:
"""
logger.info("## Create a Completion")

response = watsonx_llm.complete("What is a Generative AI?")
logger.debug(response)

"""
From the `CompletionResponse`, you can also retrieve a raw response returned by the service:
"""
logger.info("From the `CompletionResponse`, you can also retrieve a raw response returned by the service:")

logger.debug(response.raw)

"""
You can also call a model that provides a prompt template:
"""
logger.info("You can also call a model that provides a prompt template:")


template = "What is {object} and how does it work?"
prompt_template = PromptTemplate(template=template)

prompt = prompt_template.format(object="a loan")

response = watsonx_llm.complete(prompt)
logger.debug(response)

"""
## Calling `chat` with a list of messages
Create `chat` completions by providing a list of messages:
"""
logger.info("## Calling `chat` with a list of messages")


messages = [
    ChatMessage(role="system", content="You are an AI assistant"),
    ChatMessage(role="user", content="Who are you?"),
]
response = watsonx_llm.chat(
    messages, max_new_tokens=20, decoding_method="greedy"
)
logger.debug(response)

"""
Note that we changed the `max_new_tokens` parameter to `20` and the `decoding_method` parameter to `greedy`.

## Streaming the model output 

Stream the model's response:
"""
logger.info("## Streaming the model output")

for chunk in watsonx_llm.stream_complete(
    "Describe your favorite city and why it is your favorite."
):
    logger.debug(chunk.delta, end="")

"""
Similarly, to stream the `chat` completions, use the following code:
"""
logger.info("Similarly, to stream the `chat` completions, use the following code:")

messages = [
    ChatMessage(role="system", content="You are an AI assistant"),
    ChatMessage(role="user", content="Who are you?"),
]

for chunk in watsonx_llm.stream_chat(messages):
    logger.debug(chunk.delta, end="")

logger.info("\n\n[DONE]", bright=True)