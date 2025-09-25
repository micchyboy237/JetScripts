from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AmazonBedrockModel
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
id: amazon-bedrock
title: Amazon Bedrock
sidebar_label: Amazon Bedrock
---

DeepEval supports using any Amazon Bedrock model for all evaluation metrics. To get started, you'll need to set up your AWS credentials.

### Setting Up Your API Key

To use Amazon Bedrock for `deepeval`'s LLM-based evaluations (metrics evaluated using an LLM), provide your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the CLI:
"""
logger.info("### Setting Up Your API Key")

export AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>

"""
Alternatively, if you're working in a notebook environment (e.g., Jupyter or Colab), set your keys in a cell:
"""
logger.info("Alternatively, if you're working in a notebook environment (e.g., Jupyter or Colab), set your keys in a cell:")

%env AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
%env AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>

"""
### Python

To use Amazon bedrock models for DeepEval metrics, define an `AmazonBedrockModel` and specify the model you want to use.
"""
logger.info("### Python")


model = AmazonBedrockModel(
    model_id="anthropic.claude-3-opus-20240229-v1:0",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)

"""
There are **TWO** mandatory and **FIVE** optional parameters when creating an `AmazonBedrockModel`:

- `model_id`: A string specifying the bedrock model identifier to call (e.g. `anthropic.claude-3-opus-20240229-v1:0`).
- `region_name`: A string specifying the AWS region hosting your Bedrock endpoint (e.g. `us-east-1`).
- [Optional] `aws_access_key_id`: A string specifiying your AWS Access Key ID. If omitted, falls back to the default AWS credentials chain.
- [Optional] `aws_secret_access_key`: A string specifiying your AWS Secret Access Key. If omitted, falls back to the default AWS credentials chain.
- [Optional] `temperature`: A float specifying the model temperature. Defaulted to 0.
- [Optional] `input_token_cost`: A float specifying the per-input-token cost in USD. Defaulted to 0.
- [Optional] `output_token_cost`: A float specifying the per-output-token cost in USD. Defaulted to 0.
- [Optional] `generation_kwargs`: A dictionary of additional generation parameters supported by your model provider.

:::tip
Any `**kwargs` you would like to use for your model can be passed through the `generation_kwargs` parameter. However, we request you to double check the params supported by the model and your model provider in their [official docs](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-parameters.html).
:::

### Available Amazon Bedrock Models

:::note
This list only displays some of the available models. For a comprehensive list, refer to the Amazon Bedrock's official documentation.
:::

Below is a list of commonly used Amazon Bedrock foundation models:

- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `amazon.titan-text-express-v1`
- `amazon.titan-text-premier-v1:0`
- `amazon.nova-micro-v1:0`
- `amazon.nova-lite-v1:0`
- `amazon.nova-pro-v1:0`
- `amazon.nova-premier-v1:0`
"""
logger.info("### Available Amazon Bedrock Models")

logger.info("\n\n[DONE]", bright=True)