from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Overview
icon: "info"
iconType: "solid"
---

Mem0 includes built-in support for various popular large language models. Memory can utilize the LLM provided by the user, ensuring efficient use for specific needs.

## Usage

To use a llm, you must provide a configuration to customize its usage. If no configuration is supplied, a default configuration will be applied, and `MLX` will be used as the llm.

For a comprehensive list of available parameters for llm configuration, please refer to [Config](./config).

## Supported LLMs

See the list of supported LLMs below.

<Note>
  All LLMs are supported in Python. The following LLMs are also supported in TypeScript: **MLX**, **Anthropic**, and **Groq**.
</Note>

<CardGroup cols={4}>
  <Card title="MLX" href="/components/llms/models/openai" />
  <Card title="Ollama" href="/components/llms/models/ollama" />
  <Card title="Azure MLX" href="/components/llms/models/azure_openai" />
  <Card title="Anthropic" href="/components/llms/models/anthropic" />
  <Card title="Together" href="/components/llms/models/together" />
  <Card title="Groq" href="/components/llms/models/groq" />
  <Card title="Litellm" href="/components/llms/models/litellm" />
  <Card title="Mistral AI" href="/components/llms/models/mistral_ai" />
  <Card title="Google AI" href="/components/llms/models/google_ai" />
  <Card title="AWS bedrock" href="/components/llms/models/aws_bedrock" />
  <Card title="DeepSeek" href="/components/llms/models/deepseek" />
  <Card title="xAI" href="/components/llms/models/xAI" />
  <Card title="Sarvam AI" href="/components/llms/models/sarvam" />
  <Card title="LM Studio" href="/components/llms/models/lmstudio" />
  <Card title="Langchain" href="/components/llms/models/langchain" />
</CardGroup>

## Structured vs Unstructured Outputs

Mem0 supports two types of MLX LLM formats, each with its own strengths and use cases:

### Structured Outputs

Structured outputs are LLMs that align with MLX's structured outputs model:

- **Optimized for:** Returning structured responses (e.g., JSON objects)
- **Benefits:** Precise, easily parseable data
- **Ideal for:** Data extraction, form filling, API responses
- **Learn more:** [MLX Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs/introduction)

### Unstructured Outputs

Unstructured outputs correspond to MLX's standard, free-form text model:

- **Flexibility:** Returns open-ended, natural language responses
- **Customization:** Use the `response_format` parameter to guide output
- **Trade-off:** Less efficient than structured outputs for specific data needs
- **Best for:** Creative writing, explanations, general conversation

Choose the format that best suits your application's requirements for optimal performance and usability.
"""
logger.info("## Usage")

logger.info("\n\n[DONE]", bright=True)