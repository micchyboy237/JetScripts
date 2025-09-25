from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from jet.logger import logger
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
---
# id: vertex-ai
title: Vertex AI
sidebar_label: Vertex AI
---

You can also use Google Cloud's Vertex AI models, including Gemini or your own fine-tuned models, with DeepEval.

:::info
To use Vertex AI, you must have the following:

1. A Google Cloud project with the Vertex AI API enabled
2. Application Default Credentials set up:
"""
logger.info("# id: vertex-ai")

gcloud auth application-default login

"""
:::

### Command Line

Run the following command in your terminal to configure your deepeval environment to use Gemini models through Vertex AI for all metrics.
"""
logger.info("### Command Line")

deepeval set-gemini \
    --model-name=<model_name> \ # e.g. "gemini-2.0-flash-001"
    --project-id=<project_id> \
    --location=<location> # e.g. "us-central1"

"""
:::info
The CLI command above sets Gemini (via Vertex AI) as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Gemini:
"""
logger.info("The CLI command above sets Gemini (via Vertex AI) as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Gemini:")

deepeval unset-gemini

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

Alternatively, you can specify your model directly in code using `GeminiModel` from DeepEval's model collection. By default, `model_name` is set to `gemini-1.5-pro`.
"""
logger.info("### Python")


model = GeminiModel(
    model_name="gemini-1.5-pro",
    project="Your Project ID",
    location="us-central1",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **THREE** mandatory and **ONE** optional parameters when creating an `GeminiModel` through Vertex AI:

- `model_name`: A string specifying the name of the Gemini model to use.
- `project`: A string specifying your Google Cloud project ID.
- `location`: A string specifying the Google Cloud location.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters).
:::

### Available Vertex AI Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Vertex AI's official documentation.
:::

Below is a list of commonly used Gemini models:

`gemini-2.0-pro-exp-02-05`  
`gemini-2.0-flash`  
`gemini-2.0-flash-001`  
`gemini-2.0-flash-002`  
`gemini-2.0-flash-lite`  
`gemini-2.0-flash-lite-001`  
`gemini-1.5-pro`  
`gemini-1.5-pro-001`  
`gemini-1.5-pro-002`  
`gemini-1.5-flash`  
`gemini-1.5-flash-001`  
`gemini-1.5-flash-002`  
`gemini-1.0-pro`  
`gemini-1.0-pro-001`  
`gemini-1.0-pro-002`  
`gemini-1.0-pro-vision`  
`gemini-1.0-pro-vision-001`
"""
logger.info("### Available Vertex AI Models")

logger.info("\n\n[DONE]", bright=True)