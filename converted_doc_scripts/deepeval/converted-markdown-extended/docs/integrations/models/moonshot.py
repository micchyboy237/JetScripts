from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import KimiModel
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
# id: moonshot
title: Moonshot
sidebar_label: Moonshot
---

DeepEval's integration with Moonshot AI allows you to use any Moonshot models to power all of DeepEval's metrics.

### Command Line

To configure your Moonshot model through the CLI, run the following command:
"""
logger.info("# id: moonshot")

deepeval set-moonshot \
    --model "kimi-k2-0711-preview" \
    --api-key "your-api-key" \
    --temperature=0

"""
:::info
The CLI command above sets Moonshot as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Moonshot:
"""
logger.info("The CLI command above sets Moonshot as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Moonshot:")

deepeval unset-moonshot

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

Alternatively, you can define `KimiModel` directly in python code:
"""
logger.info("### Python")


model = KimiModel(
    model_name="kimi-k2-0711-preview",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** mandatory and **ONE** optional parameters when creating an `KimiModel`:

- `model`: A string specifying the name of the Kimi model to use.
- `api_key`: A string specifying your Kimi API key for authentication.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://docs.together.ai/docs/inference-parameters).
:::

### Available Moonshot Models

Below is a comprehensive list of available Moonshot models:

- `kimi-k2-0711-preview`
- `kimi-thinking-preview`
- `moonshot-v1-8k`
- `moonshot-v1-32k`
- `moonshot-v1-128k`
- `moonshot-v1-8k-vision-preview`
- `moonshot-v1-32k-vision-preview`
- `moonshot-v1-128k-vision-preview`
- `kimi-latest-8k`
- `kimi-latest-32k`
- `kimi-latest-128k`
"""
logger.info("### Available Moonshot Models")

logger.info("\n\n[DONE]", bright=True)