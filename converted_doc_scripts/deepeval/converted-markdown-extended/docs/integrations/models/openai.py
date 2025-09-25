from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GPTModel
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
# id: ollama
title: Ollama
sidebar_label: Ollama
---

By default, DeepEval uses `gpt-4.1` to power all of its evaluation metrics. To enable this, you’ll need to set up your Ollama API key. DeepEval also supports all other Ollama models, which can be configured directly in Python.

### Setting Up Your API Key

DeepEval autoloads `.env.local` then `.env` at import time (process env -> `.env.local` -> `.env`).

**Recommended (local dev):**
"""
logger.info("# id: ollama")

# OPENAI_API_KEY=<your-ollama-api-key>

"""
Alternative (Shell/CI)
"""
logger.info("Alternative (Shell/CI)")

# export OPENAI_API_KEY=<your-ollama-api-key>

"""
Alternative (notebook)

# If you're working in a notebook environment (Jupyter or Colab), set your `OPENAI_API_KEY` in a cell:
"""
logger.info("Alternative (notebook)")

# %env OPENAI_API_KEY=<your-ollama-api-key>

"""
### Command Line

Run the following command in your CLI to specify an Ollama model to power all metrics.
"""
logger.info("### Command Line")

deepeval set-ollama \
    --model=gpt-4.1
    --cost_per_input_token=0.000002
    --cost_per_output_token=0.000008

"""
:::info
The CLI command above sets `gpt-4.1` as the default model for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset the current settings:
"""
logger.info("The CLI command above sets `gpt-4.1` as the default model for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset the current settings:")

deepeval unset-ollama

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

You may use Ollama models other than `gpt-4.1`, which can be configured directly in python code through DeepEval's `GPTModel`.

:::info
You may want to use stronger reasoning models like `gpt-4.1` for metrics that require a high level of reasoning — for example, a custom GEval for mathematical correctness.
:::
"""
logger.info("### Python")


model = GPTModel(
    model="llama3.2",
    temperature=0,
    cost_per_input_token=0.000002,
    cost_per_output_token=0.000008
)
answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **ONE** mandatory and **ONE** optional parameters when creating a `GPTModel`:

- `model`: A string specifying the name of the GPT model to use. Defaulted to `gpt-4o`.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `cost_per_input_token`: A float specifying the cost for each input token for the provided model.
- [Optional] `cost_per_output_token`: A float specifying the cost for each output token for the provided model.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://platform.ollama.com/docs/api-reference/responses/create).
:::

### Available Ollama Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Ollama's official documentation.
:::

Below is a list of commonly used Ollama models:

- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-4.1`
- `gpt-4.5-preview`
- `gpt-4o`
- `llama3.2`
- `o1`
- `o1-pro`
- `o1-mini`
- `o3-mini`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-4-32k`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-instruct`
- `gpt-3.5-turbo-16k-0613`
- `davinci-002`
- `babbage-002`
"""
logger.info("### Available Ollama Models")

logger.info("\n\n[DONE]", bright=True)