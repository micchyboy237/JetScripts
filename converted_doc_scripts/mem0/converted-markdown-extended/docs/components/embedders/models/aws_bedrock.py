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

To use AWS Bedrock embedding models, you need to have the appropriate AWS credentials and permissions. The embeddings implementation relies on the `boto3` library.

### Setup
- Ensure you have model access from the [AWS Bedrock Console](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)
- Authenticate the boto3 client using a method described in the [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
- Set up environment variables for authentication:
  ```bash
  export AWS_REGION=us-east-1
  export AWS_ACCESS_KEY_ID=your-access-key
  export AWS_SECRET_ACCESS_KEY=your-secret-key
  ```

### Usage

<CodeGroup>
"""
logger.info("### Setup")


# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

os.environ["AWS_REGION"] = "us-west-2"
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"

config = {
    "embedder": {
        "provider": "aws_bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v2:0"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice")

"""
</CodeGroup>

### Config

Here are the parameters available for configuring AWS Bedrock embedder:

<Tabs>
<Tab title="Python">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `model` | The name of the embedding model to use | `amazon.titan-embed-text-v1` |
</Tab>
</Tabs>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)