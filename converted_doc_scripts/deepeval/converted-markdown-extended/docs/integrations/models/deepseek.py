from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepSeekModel
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
# id: deepseek
title: DeepSeek
sidebar_label: DeepSeek
---

DeepEval allows you to use `deepseek-chat` and `deepseek-reasoner` directly from DeepSeek to run all of DeepEval's metrics, which can be set through the CLI or in python.

### Command Line

To configure your DeepSeek model through the CLI, run the following command:
"""
logger.info("# id: deepseek")

deepeval set-deepseek --model deepseek-chat \
    --api-key="your-api-key" \
    --temperature=0

"""
The CLI command above sets `deepseek-chat` as the default model for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset DeepSeek:
"""
logger.info("The CLI command above sets `deepseek-chat` as the default model for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset DeepSeek:")

deepeval unset-deepseek

"""
:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

You can also specify your model directly in code using `DeepSeekModel`.
"""
logger.info("### Python")


model = DeepSeekModel(
    model="deepseek-chat",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** mandatory and **ONE** optional parameters when creating a `DeepSeekModel`:

- `model`: A string specifying the name of the DeepSeek model to use. Either be `deepseek-chat` or `deepseek-reasoner`.
- `api_key`: A string specifying your DeepSeek API key for authentication.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://api-docs.deepseek.com/api/create-chat-completion#request).
:::

### Available DeepSeek Models

Below is the comprehensive list of available DeepSeek models in DeepEval:

- `deepseek-chat`
- `deepseek-reasoner`
"""
logger.info("### Available DeepSeek Models")

logger.info("\n\n[DONE]", bright=True)