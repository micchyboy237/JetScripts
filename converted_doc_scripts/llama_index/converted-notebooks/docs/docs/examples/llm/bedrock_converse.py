from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock_converse import BedrockConverse
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/bedrock_converse.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Bedrock Converse

## Basic Usage

#### Call `complete` with a prompt

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Bedrock Converse")

# %pip install llama-index-llms-bedrock-converse

# !pip install llama-index


profile_name = "Your aws profile name"
resp = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
).complete("Paul Graham is ")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
).chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
)
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
)
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
## Connect to Bedrock with Access Keys
"""
logger.info("## Connect to Bedrock with Access Keys")


llm = BedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    region_name="AWS Region to use, eg. us-east-1",
)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
## Function Calling

Claude, Command and Mistral Large models supports native function calling through AWS Bedrock Converse. There's a seamless integration with LlamaIndex tools, through the `predict_and_call` function on the `llm`. 

This allows the user to attach any tools and let the LLM decide which tools to call (if any).

If you wish to perform tool calling as part of an agentic loop, check out our [agent guides](https://docs.llamaindex.ai/en/latest/module_guides/deploying/agents/) instead.

**NOTE**: Not all models from AWS Bedrock support function calling and the Converse API. [Check the available features of each LLM here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-features.html).
"""
logger.info("## Function Calling")



def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def mystery(a: int, b: int) -> int:
    """Mystery function on two integers."""
    return a * b + a + b


mystery_tool = FunctionTool.from_defaults(fn=mystery)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    profile_name=profile_name,
)

response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg="What happens if I run the mystery function on 5 and 7",
)

logger.debug(str(response))

response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg=(
        """What happens if I run the mystery function on the following pairs of numbers? Generate a separate result for each row:
- 1 and 2
- 8 and 4
- 100 and 20

NOTE: you need to run the mystery function for all of the pairs above at the same time \
"""
    ),
    allow_parallel_tool_calls=True,
)

logger.debug(str(response))

for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
## Async
"""
logger.info("## Async")


llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    region_name="AWS Region to use, eg. us-east-1",
)
resp = llm.complete("Paul Graham is ")
logger.success(format_json(resp))

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)