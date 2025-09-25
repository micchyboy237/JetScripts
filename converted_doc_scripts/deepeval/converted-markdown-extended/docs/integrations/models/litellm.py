from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import LiteLLMModel
from jet.logger import logger
from pydantic import BaseModel
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
# id: litellm
title: LiteLLM
sidebar_label: LiteLLM
---

DeepEval allows you to use any model supported by LiteLLM to run evals, either through the CLI or directly in Python.

:::note
Before getting started, make sure you have LiteLLM installed. It will not be installed automatically with DeepEval, you need to install it separately:
"""
logger.info("# id: litellm")

pip install litellm

"""
:::

### Command Line

To configure your LiteLLM model through the CLI, run the following command. You must specify the provider in the model name:
"""
logger.info("### Command Line")

deepeval set-litellm ollama/gpt-3.5-turbo

deepeval set-litellm anthropic/claude-3-opus

deepeval set-litellm google/gemini-pro

"""
You can also specify additional parameters:
"""
logger.info("You can also specify additional parameters:")

deepeval set-litellm ollama/gpt-3.5-turbo --api-key="your-api-key"

deepeval set-litellm ollama/gpt-3.5-turbo --api-base="https://your-custom-endpoint.com"

deepeval set-litellm ollama/gpt-3.5-turbo \
    --api-key="your-api-key" \
    --api-base="https://your-custom-endpoint.com"

"""
:::info
The CLI command above sets LiteLLM as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset LiteLLM:
"""
logger.info("The CLI command above sets LiteLLM as the default provider for all metrics, unless overridden in Python code. To use a different default model provider, you must first unset LiteLLM:")

deepeval unset-litellm

"""
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Python

When using LiteLLM in Python, you must always specify the provider in the model name. Here's how to use `LiteLLMModel` from DeepEval's model collection:
"""
logger.info("### Python")


model = LiteLLMModel(
    model="ollama/gpt-3.5-turbo",  # Provider must be specified
    # optional, can be set via environment variable
    api_base="your-api-base",  # optional, for custom endpoints
    temperature=0
)

answer_relevancy = AnswerRelevancyMetric(model=model)

"""
The `LiteLLMModel` class accepts the following parameters:

- `model` (required): A string specifying the provider and model name (e.g., "ollama/gpt-3.5-turbo", "anthropic/claude-3-opus")
- `api_key` (optional): A string specifying the API key for the model
- `api_base` (optional): A string specifying the base URL for the model API
- `temperature` (optional): A float specifying the model temperature. Defaults to 0
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://docs.litellm.ai/docs/providers/custom_llm_server).
:::

### Environment Variables

You can also configure LiteLLM using environment variables:
"""
logger.info("### Environment Variables")

# export OPENAI_API_KEY="your-api-key"

# export ANTHROPIC_API_KEY="your-api-key"

export GOOGLE_API_KEY="your-api-key"

export LITELLM_API_BASE="https://your-custom-endpoint.com"

"""
### Available Models

:::note
This list only displays some of the available models. For a complete list of supported models and their capabilities, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).
:::

#### Ollama Models
- `ollama/gpt-3.5-turbo`
- `ollama/gpt-4`
- `ollama/gpt-4-turbo-preview`

#### Ollama Models
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`

#### Google Models
- `google/gemini-pro`
- `google/gemini-ultra`

#### Mistral Models
- `mistral/mistral-small`
- `mistral/mistral-medium`
- `mistral/mistral-large`

#### LM Studio Models
- `lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF`
- `lm-studio/Mistral-7B-Instruct-v0.2-GGUF`
- `lm-studio/Phi-2-GGUF`

#### Ollama Models
- `ollama/llama2`
- `ollama/mistral`
- `ollama/codellama`
- `ollama/neural-chat`
- `ollama/starling-lm`

:::note
When using LM Studio, you need to specify the API base URL. By default, LM Studio runs on `http://localhost:1234/v1`.

When using Ollama, you need to specify the API base URL. By default, Ollama runs on `http://localhost:11434/v1`.
:::

### Examples

#### Basic Usage with Different Providers
"""
logger.info("### Available Models")


model = LiteLLMModel(model="ollama/gpt-3.5-turbo")
metric = AnswerRelevancyMetric(model=model)

model = LiteLLMModel(model="anthropic/claude-3-opus")
metric = AnswerRelevancyMetric(model=model)

model = LiteLLMModel(model="google/gemini-pro")
metric = AnswerRelevancyMetric(model=model)

model = LiteLLMModel(
    model="lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF",
    api_base="http://localhost:1234/v1",  # LM Studio default URL
      # LM Studio uses a fixed API key
)
metric = AnswerRelevancyMetric(model=model)

model = LiteLLMModel(
    model="ollama/llama2",
    api_base="http://localhost:11434/v1",  # Ollama default URL
      # Ollama uses a fixed API key
)
metric = AnswerRelevancyMetric(model=model)

"""
#### Using Custom Endpoint
"""
logger.info("#### Using Custom Endpoint")

model = LiteLLMModel(
    model="custom/your-model-name",  # Provider must be specified
    api_base="https://your-custom-endpoint.com"
)

"""
#### Using with Schema Validation
"""
logger.info("#### Using with Schema Validation")


class ResponseSchema(BaseModel):
    score: float
    reason: str

model = LiteLLMModel(model="ollama/gpt-3.5-turbo")
response, cost = model.generate(
    "Rate this answer: 'The capital of France is Paris'",
    schema=ResponseSchema
)

model = LiteLLMModel(
    model="lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF",
    api_base="http://localhost:1234/v1"
)
response, cost = model.generate(
    "Rate this answer: 'The capital of France is Paris'",
    schema=ResponseSchema
)

model = LiteLLMModel(
    model="ollama/llama2",
    api_base="http://localhost:11434/v1"
)
response, cost = model.generate(
    "Rate this answer: 'The capital of France is Paris'",
    schema=ResponseSchema
)

"""
### Best Practices

1. **Provider Specification**: Always specify the provider in the model name (e.g., "ollama/gpt-3.5-turbo", "anthropic/claude-3-opus", "lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF", "ollama/llama2")

2. **API Key Security**: Store your API keys in environment variables rather than hardcoding them in your scripts.

3. **Model Selection**: Choose the appropriate model based on your needs:
   - For simple tasks: Use smaller models like `ollama/gpt-3.5-turbo`
   - For complex reasoning: Use larger models like `ollama/gpt-4` or `anthropic/claude-3-opus`
   - For cost-sensitive applications: Use models like `mistral/mistral-small` or `anthropic/claude-3-haiku`
   - For local development: 
     - Use LM Studio models like `lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF`
     - Use Ollama models like `ollama/llama2` or `ollama/mistral`

4. **Error Handling**: Implement proper error handling for API rate limits and connection issues.

5. **Cost Management**: Monitor your usage and costs, especially when using larger models.

6. **Local Model Setup**:
   - **LM Studio**:
     - Make sure LM Studio is running and the model is loaded
     - Use the correct API base URL (default: `http://localhost:1234/v1`)
     - Use the fixed API key "lm-studio"
     - Ensure the model is properly downloaded and loaded in LM Studio
   
   - **Ollama**:
     - Make sure Ollama is running and the model is pulled
     - Use the correct API base URL (default: `http://localhost:11434/v1`)
     - Use the fixed API key "ollama"
     - Pull the model first using `ollama pull llama2` (or your chosen model)
     - Ensure you have enough system resources for the model
"""
logger.info("### Best Practices")

logger.info("\n\n[DONE]", bright=True)