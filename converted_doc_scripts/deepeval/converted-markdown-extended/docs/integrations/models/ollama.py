from deepeval.metrics import AnswerRelevancyMetric
from jet.adapters.haystack.deepeval.ollama_model import OllamaModel
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

DeepEval allows you to use any model from Ollama to run evals, either through the CLI or directly in python.

:::note
Before getting started, make sure your Ollama model is installed and running. See the full list of available models [here](<](https://ollama.com/search)>).
"""
logger.info("# id: ollama")

ollama run deepseek-r1:1.5b

"""
:::

### Environment Setup
DeepEval can use a local Ollama server (default: `http://127.0.0.1:11434`).
Optionally set a custom host:
"""
logger.info("### Environment Setup")

OLLAMA_HOST=http://127.0.0.1:11434

"""
### Command Line

To configure your Ollama model through the CLI, run the following command. Replace `deepseek-r1:1.5b` with any Ollama-supported model of your choice.
"""
logger.info("### Command Line")

deepeval set-ollama deepseek-r1:1.5b

"""
You may also specify the **base URL** of your local Ollama model instance if you've defined a custom port. By default, the base URL is set to `http://localhost:11434`.
"""
logger.info("You may also specify the **base URL** of your local Ollama model instance if you've defined a custom port. By default, the base URL is set to `http://localhost:11434`.")

deepeval set-ollama deepseek-r1:1.5b \
    --base-url="http://localhost:11434"

"""
:::info
The CLI command above sets Ollama as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Ollama:
"""
logger.info("The CLI command above sets Ollama as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset Ollama:")

deepeval unset-ollama

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

Alternatively, you can specify your model directly in code using `OllamaModel` from DeepEval's model collection.
"""
logger.info("### Python")


model = OllamaModel(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **FIVE** mandatory and **ONE** optional parameters when creating an `AzureOpenAIModel`:

- `model`: A string specifying the name of the Ollama model to use.
- [Optional] `base_url`: A string specifying the base URL of the Ollama server. Defaulted to `'http://localhost:11434'`.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://ollama.readthedocs.io/en/api/#parameters).
:::

### Available Ollama Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Ollama's official documentation.
:::

Below is a list of commonly used Ollama models:

- `deepseek-r1`
- `llama3.1`
- `gemma`
- `qwen`
- `mistral`
- `codellama`
- `phi3`
- `tinyllama`
- `starcoder2`
"""
logger.info("### Available Ollama Models")

logger.info("\n\n[DONE]", bright=True)