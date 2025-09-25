from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AnthropicModel
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
# id: anthropic
title: Ollama
sidebar_label: Ollama
---

DeepEval supports using any Ollama model for all evaluation metrics. To get started, you'll need to set up your Ollama API key.

### Setting Up Your API Key

# To use Ollama for `deepeval`'s LLM-based evaluations (metrics evaluated using an LLM), provide your `ANTHROPIC_API_KEY` in the CLI:
"""
logger.info("# id: anthropic")

# export ANTHROPIC_API_KEY=<your-anthropic-api-key>

"""
# Alternatively, if you're working in a notebook environment (e.g., Jupyter or Colab), set your `ANTHROPIC_API_KEY` in a cell:
"""
# logger.info("Alternatively, if you're working in a notebook environment (e.g., Jupyter or Colab), set your `ANTHROPIC_API_KEY` in a cell:")

# %env ANTHROPIC_API_KEY=<your-anthropic-api-key>

"""
### Python

To use Ollama models for DeepEval metrics, define an `AnthropicModel` and specify the model you want to use. By default, the `model` is set to `claude-3-7-sonnet-latest`.
"""
logger.info("### Python")


model = AnthropicModel(
    model="claude-3-7-sonnet-latest",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** optional parameters when creating an `AnthropicModel`:

- [Optional] `model`: A string specifying which of Ollama's Claude models to use. Defaulted to `'claude-3-7-sonnet-latest'`.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://docs.anthropic.com/en/docs/about-claude/models/overview).
:::

### Available Ollama Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Ollama's official documentation.
:::

Below is a list of commonly used Ollama models:

- `claude-3-7-sonnet-latest`
- `claude-3-5-haiku-latest`
- `llama3.2`
- `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-instant-1.2`
"""
logger.info("### Available Ollama Models")

logger.info("\n\n[DONE]", bright=True)