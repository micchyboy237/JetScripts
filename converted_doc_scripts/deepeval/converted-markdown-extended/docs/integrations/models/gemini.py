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
# id: gemini
title: Gemini
sidebar_label: Gemini
---

DeepEval allows you to directly integrate Gemini models into all available LLM-based metrics, either through the command line or directly within your python code.

### Command Line

Run the following command in your terminal to configure your deepeval environment to use Gemini models for all metrics.
"""
logger.info("# id: gemini")

deepeval set-gemini \
    --model-name=<model_name> \ # e.g. "gemini-2.0-flash-001"
    --google-api-key=<api_key>

"""
:::info
The CLI command above sets Gemini as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Gemini:
"""
logger.info("The CLI command above sets Gemini as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Gemini:")

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
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** mandatory and **ONE** optional parameters when creating an `GeminiModel`:

- `model_name`: A string specifying the name of the Gemini model to use.
- `api_key`: A string specifying the Google API key for authentication.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://ai.google.dev/api/generate-content#generationconfig).
:::

### Available Gemini Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Gemini's official documentation.
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
logger.info("### Available Gemini Models")

logger.info("\n\n[DONE]", bright=True)