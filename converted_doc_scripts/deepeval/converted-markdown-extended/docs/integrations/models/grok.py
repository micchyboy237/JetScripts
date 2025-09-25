from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GrokModel
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
# id: grok
title: Grok
sidebar_label: Grok
---

DeepEval allows you to use any Grok model from xAI to run evals, either through the CLI or directly in python.

:::info
To use Grok, you must first install the xAI SDK:
"""
logger.info("# id: grok")

pip install xai-sdk

"""
:::

### Command Line

To configure Grok through the CLI, run the following command:
"""
logger.info("### Command Line")

deepeval set-grok --model grok-4-0709 \
    --api-key="your-api-key" \
    --temperature=0

"""
The CLI command above sets the specified Grok model as the default llm-judge for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Grok:
"""
logger.info("The CLI command above sets the specified Grok model as the default llm-judge for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Grok:")

deepeval unset-grok

"""
:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

Alternatively, you can specify your model directly in code using `GrokModel` from DeepEval's model collection.
"""
logger.info("### Python")


model = GrokModel(
    model_name="grok-4-0709",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** mandatory and **ONE** optional parameters when creating an `GrokModel`:

- `model`: A string specifying the name of the Grok model to use.
- [Optional] `api_key`: A string specifying your Grok API key for authentication.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://docs.x.ai/docs/guides/function-calling#function-calling-modes).
:::

### Available Grok Models

Below is the comprehensive list of available Grok models in DeepEval:

- `grok-4-0709`
- `grok-3`
- `grok-3-mini`
- `grok-3-fast`
- `grok-3-mini-fast`
- `grok-2-vision-1212`
"""
logger.info("### Available Grok Models")

logger.info("\n\n[DONE]", bright=True)