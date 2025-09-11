from jet.logger import logger
from langchain_community.llms import Databricks
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
# Databricks

> [Databricks](https://www.databricks.com/) Lakehouse Platform unifies data, analytics, and AI on one platform.


This notebook provides a quick overview for getting started with Databricks [LLM models](https://python.langchain.com/docs/concepts/text_llms). For detailed documentation of all features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.databricks.Databricks.html).

## Overview

`Databricks` LLM class wraps a completion endpoint hosted as either of these two endpoint types:

* [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html), recommended for production and development,
* Cluster driver proxy app, recommended for interactive development.

This example notebook shows how to wrap your LLM endpoint and use it as an LLM in your LangChain application.

## Limitations

The `Databricks` LLM class is *legacy* implementation and has several limitations in the feature compatibility.

* Only supports synchronous invocation. Streaming or async APIs are not supported.
* `batch` API is not supported.

To use those features, please use the new [ChatDatabricks](https://python.langchain.com/docs/integrations/chat/databricks) class instead. `ChatDatabricks` supports all APIs of `ChatModel` including streaming, async, batch, etc.

## Setup

To access Databricks models you'll need to create a Databricks account, set up credentials (only if you are outside Databricks workspace), and install required packages.

### Credentials (only if you are outside Databricks)

If you are running LangChain app inside Databricks, you can skip this step.

Otherwise, you need manually set the Databricks workspace hostname and personal access token to `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively. See [Authentication Documentation](https://docs.databricks.com/en/dev-tools/auth/index.html#databricks-personal-access-tokens) for how to get an access token.
"""
logger.info("# Databricks")

# import getpass

os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
if "DATABRICKS_TOKEN" not in os.environ:
#     os.environ["DATABRICKS_TOKEN"] = getpass.getpass(
        "Enter your Databricks access token: "
    )

"""
Alternatively, you can pass those parameters when initializing the `Databricks` class.
"""
logger.info("Alternatively, you can pass those parameters when initializing the `Databricks` class.")


databricks = Databricks(
    host="https://your-workspace.cloud.databricks.com",
    token=dbutils.secrets.get(scope="YOUR_SECRET_SCOPE", key="databricks-token"),  # noqa: F821
)

"""
### Installation

The LangChain Databricks integration lives in the `langchain-community` package. Also, `mlflow >= 2.9 ` is required to run the code in this notebook.
"""
logger.info("### Installation")

# %pip install -qU langchain-community mlflow>=2.9.0

"""
## Wrapping Model Serving Endpoint

### Prerequisites:

* An LLM was registered and deployed to [a Databricks serving endpoint](https://docs.databricks.com/machine-learning/model-serving/index.html).
* You have ["Can Query" permission](https://docs.databricks.com/security/auth-authz/access-control/serving-endpoint-acl.html) to the endpoint.

The expected MLflow model signature is:

  * inputs: `[{"name": "prompt", "type": "string"}, {"name": "stop", "type": "list[string]"}]`
  * outputs: `[{"type": "string"}]`

### Invocation
"""
logger.info("## Wrapping Model Serving Endpoint")


llm = Databricks(endpoint_name="YOUR_ENDPOINT_NAME")
llm.invoke("How are you?")

llm.invoke("How are you?", stop=["."])

"""
### Transform Input and Output

Sometimes you may want to wrap a serving endpoint that has imcompatible model signature or you want to insert extra configs. You can use the `transform_input_fn` and `transform_output_fn` arguments to define additional pre/post process.
"""
logger.info("### Transform Input and Output")

def transform_input(**request):
    full_prompt = f"""{request["prompt"]}
    Be Concise.
    """
    request["prompt"] = full_prompt
    return request


def transform_output(response):
    return response.upper()


llm = Databricks(
    endpoint_name="YOUR_ENDPOINT_NAME",
    transform_input_fn=transform_input,
    transform_output_fn=transform_output,
)

llm.invoke("How are you?")

logger.info("\n\n[DONE]", bright=True)