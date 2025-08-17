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

Mem0 offers support for various embedding models, allowing users to choose the one that best suits their needs.

## Supported Embedders

See the list of supported embedders below.

<Note>
  The following embedders are supported in the Python implementation. The TypeScript implementation currently only supports MLX.
</Note>

<CardGroup cols={4}>
  <Card title="MLX" href="/components/embedders/models/openai"></Card>
  <Card title="Azure MLX" href="/components/embedders/models/azure_openai"></Card>
  <Card title="Ollama" href="/components/embedders/models/ollama"></Card>
  <Card title="Hugging Face" href="/components/embedders/models/huggingface"></Card>
  <Card title="Google AI" href="/components/embedders/models/google_AI"></Card>
  <Card title="Vertex AI" href="/components/embedders/models/vertexai"></Card>
  <Card title="Together" href="/components/embedders/models/together"></Card>
  <Card title="LM Studio" href="/components/embedders/models/lmstudio"></Card>
  <Card title="Langchain" href="/components/embedders/models/langchain"></Card>
  <Card title="AWS Bedrock" href="/components/embedders/models/aws_bedrock"></Card>
</CardGroup>

## Usage

To utilize a embedder, you must provide a configuration to customize its usage. If no configuration is supplied, a default configuration will be applied, and `MLX` will be used as the embedder.

For a comprehensive list of available parameters for embedder configuration, please refer to [Config](./config).
"""
logger.info("## Supported Embedders")

logger.info("\n\n[DONE]", bright=True)