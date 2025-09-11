from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from jet.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
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
# IBM watsonx.ai

>[WatsonxLLM](https://ibm.github.io/watsonx-ai-python-sdk/fm_extensions.html#langchain) is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.

This example shows how to communicate with `watsonx.ai` models using `LangChain`.

## Overview

### Integration details
| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/ibm/) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [WatsonxLLM](https://python.langchain.com/api_reference/ibm/llms/langchain_ibm.llms.WatsonxLLM.html) | [langchain-ibm](https://python.langchain.com/api_reference/ibm/index.html) | ❌ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ibm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ibm?style=flat-square&label=%20) |

## Setup

To access IBM watsonx.ai models you'll need to create an IBM watsonx.ai account, get an API key, and install the `langchain-ibm` integration package.

### Credentials

The cell below defines the credentials required to work with watsonx Foundation Model inferencing.

**Action:** Provide the IBM Cloud user API key. For details, see
[Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
"""
logger.info("# IBM watsonx.ai")

# from getpass import getpass

# watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key

"""
Additionaly you are able to pass additional secrets as an environment variable.
"""
logger.info("Additionaly you are able to pass additional secrets as an environment variable.")


os.environ["WATSONX_URL"] = "your service instance url"
os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
os.environ["WATSONX_INSTANCE_ID"] = "your instance_id for accessing the CPD cluster"

"""
### Installation

The LangChain IBM integration lives in the `langchain-ibm` package:
"""
logger.info("### Installation")

# !pip install -qU langchain-ibm

"""
## Instantiation

You might need to adjust model `parameters` for different models or tasks. For details, refer to [documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).
"""
logger.info("## Instantiation")

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

"""
Initialize the `WatsonxLLM` class with previously set parameters.


**Note**: 

- To provide context for the API call, you must add `project_id` or `space_id`. For more information see [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
- Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas url.


You need to specify `model_id` that will be used for inferencing. All available models you can find in [documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#TextModels).
"""
logger.info("Initialize the `WatsonxLLM` class with previously set parameters.")


watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)

"""
Alternatively you can use Cloud Pak for Data credentials. For details, see [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).
"""
logger.info("Alternatively you can use Cloud Pak for Data credentials. For details, see [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).")

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)

"""
Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_tune_experiment_run.html).
"""
logger.info("Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_tune_experiment_run.html).")

watsonx_llm = WatsonxLLM(
    deployment_id="PASTE YOUR DEPLOYMENT_ID HERE",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)

"""
For certain requirements, there is an option to pass the IBM's [`APIClient`](https://ibm.github.io/watsonx-ai-python-sdk/base.html#apiclient) object into the `WatsonxLLM` class.
"""
logger.info("For certain requirements, there is an option to pass the IBM's [`APIClient`](https://ibm.github.io/watsonx-ai-python-sdk/base.html#apiclient) object into the `WatsonxLLM` class.")


api_client = APIClient(...)

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    watsonx_client=api_client,
)

"""
You can also pass the IBM's [`ModelInference`](https://ibm.github.io/watsonx-ai-python-sdk/fm_model_inference.html) object into the `WatsonxLLM` class.
"""
logger.info("You can also pass the IBM's [`ModelInference`](https://ibm.github.io/watsonx-ai-python-sdk/fm_model_inference.html) object into the `WatsonxLLM` class.")


model = ModelInference(...)

watsonx_llm = WatsonxLLM(watsonx_model=model)

"""
## Invocation
To obtain completions, you can call the model directly using a string prompt.
"""
logger.info("## Invocation")

watsonx_llm.invoke("Who is man's best friend?")

watsonx_llm.generate(
    [
        "The fastest dog in the world?",
        "Describe your chosen dog breed",
    ]
)

"""
## Streaming the Model output 

You can stream the model output.
"""
logger.info("## Streaming the Model output")

for chunk in watsonx_llm.stream(
    "Describe your favorite breed of dog and why it is your favorite."
):
    logger.debug(chunk, end="")

"""
## Chaining
Create `PromptTemplate` objects which will be responsible for creating a random question.
"""
logger.info("## Chaining")


template = "Generate a random question about {topic}: Question: "

prompt = PromptTemplate.from_template(template)

"""
Provide a topic and run the chain.
"""
logger.info("Provide a topic and run the chain.")

llm_chain = prompt | watsonx_llm

topic = "dog"

llm_chain.invoke(topic)

"""
## API reference

For detailed documentation of all `WatsonxLLM` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ibm/llms/langchain_ibm.llms.WatsonxLLM.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)