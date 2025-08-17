from jet.logger import CustomLogger
from mem0 import Memory
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
title: AWS Bedrock
---

### Setup
- Before using the AWS Bedrock LLM, make sure you have the appropriate model access from [Bedrock Console](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess).
- You will also need to authenticate the `boto3` client by using a method in the [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials)
- You will have to export `AWS_REGION`, `AWS_ACCESS_KEY`, and `AWS_SECRET_ACCESS_KEY` to set environment variables.

### Usage
"""
logger.info("### Setup")


os.environ['AWS_REGION'] = 'us-west-2'
os.environ["AWS_ACCESS_KEY_ID"] = "xx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xx"

config = {
    "llm": {
        "provider": "aws_bedrock",
        "config": {
            "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""
### Config

All available parameters for the `aws_bedrock` config are present in [Master List of All Params in Config](../config).
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)