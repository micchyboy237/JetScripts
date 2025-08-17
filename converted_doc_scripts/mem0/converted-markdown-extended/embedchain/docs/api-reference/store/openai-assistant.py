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
title: 'MLX Assistant'
---

### Arguments

<ParamField path="name" type="string">
  Name for your AI assistant
</ParamField>

<ParamField path="instructions" type="string">
  how the Assistant and model should behave or respond
</ParamField>

<ParamField path="assistant_id" type="string">
  Load existing MLX Assistant. If you pass this, you don't have to pass other arguments.
</ParamField>

<ParamField path="thread_id" type="string">
  Existing MLX thread id if exists
</ParamField>

<ParamField path="model" type="str" default="gpt-4-1106-preview">
  MLX model to use
</ParamField>

<ParamField path="tools" type="list">
  MLX tools to use. Default set to `[{"type": "retrieval"}]`
</ParamField>

<ParamField path="data_sources" type="list" default="[]">
  Add data sources to your assistant. You can add in the following format: `[{"source": "https://example.com", "data_type": "web_page"}]`
</ParamField>

<ParamField path="telemetry" type="boolean" default="True">
  Anonymous telemetry (doesn't collect any user information or user's files). Used to improve the Embedchain package utilization. Default is `True`.
</ParamField>

## Usage

For detailed guidance on creating your own MLX Assistant, click the link below. It provides step-by-step instructions to help you through the process:

<Card title="Guide to Creating Your MLX Assistant" icon="link" href="/examples/openai-assistant">
  Learn how to build an MLX Assistant using the `MLXAssistant` class.
</Card>
"""
logger.info("### Arguments")

logger.info("\n\n[DONE]", bright=True)