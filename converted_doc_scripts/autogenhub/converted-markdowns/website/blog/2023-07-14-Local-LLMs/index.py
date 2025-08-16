from autogen import oai
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
title: Use AutoGen for Local LLMs
authors: jialeliu
tags: [LLM]
---
**TL;DR:**
We demonstrate how to use autogen for local LLM application. As an example, we will initiate an endpoint using [FastChat](https://github.com/lm-sys/FastChat) and perform inference on [ChatGLMv2-6b](https://github.com/THUDM/ChatGLM2-6B).

## Preparations

### Clone FastChat

FastChat provides MLX-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for MLX APIs. However, its code needs minor modification in order to function properly.
"""
logger.info("## Preparations")

git clone https://github.com/lm-sys/FastChat.git
cd FastChat

"""
### Download checkpoint

ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. ChatGLM2-6B is its second-generation version.

Before downloading from HuggingFace Hub, you need to have Git LFS [installed](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
"""
logger.info("### Download checkpoint")

git clone https://huggingface.co/THUDM/chatglm2-6b

"""
## Initiate server

First, launch the controller
"""
logger.info("## Initiate server")

python -m fastchat.serve.controller

"""
Then, launch the model worker(s)
"""
logger.info("Then, launch the model worker(s)")

python -m fastchat.serve.model_worker --model-path chatglm2-6b

"""
Finally, launch the RESTful API server
"""
logger.info("Finally, launch the RESTful API server")

python -m fastchat.serve.openai_api_server --host localhost --port 8000

"""
Normally this will work. However, if you encounter error like [this](https://github.com/lm-sys/FastChat/issues/1641), commenting out all the lines containing `finish_reason` in `fastchat/protocol/api_protocol.py` and `fastchat/protocol/openai_api_protocol.py` will fix the problem. The modified code looks like:
"""
logger.info("Normally this will work. However, if you encounter error like [this](https://github.com/lm-sys/FastChat/issues/1641), commenting out all the lines containing `finish_reason` in `fastchat/protocol/api_protocol.py` and `fastchat/protocol/openai_api_protocol.py` will fix the problem. The modified code looks like:")

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None

class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[float] = None

"""
## Interact with model using `oai.Completion` (requires openai<1)

Now the models can be directly accessed through openai-python library as well as `autogen.oai.Completion` and `autogen.oai.ChatCompletion`.
"""
logger.info("## Interact with model using `oai.Completion` (requires openai<1)")


response = oai.Completion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "NULL", # just a placeholder
        }
    ],
    prompt="Hi",
)
logger.debug(response)

response = oai.ChatCompletion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "NULL",
        }
    ],
    messages=[{"role": "user", "content": "Hi"}]
)
logger.debug(response)

"""
If you would like to switch to different models, download their checkpoints and specify model path when launching model worker(s).

## interacting with multiple local LLMs

If you would like to interact with multiple LLMs on your local machine, replace the `model_worker` step above with a multi model variant:
"""
logger.info("## interacting with multiple local LLMs")

python -m fastchat.serve.multi_model_worker \
    --model-path lmsys/vicuna-7b-v1.3 \
    --model-names vicuna-7b-v1.3 \
    --model-path chatglm2-6b \
    --model-names chatglm2-6b

"""
The inference code would be:
"""
logger.info("The inference code would be:")


response = oai.ChatCompletion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "NULL",
        },
        {
            "model": "vicuna-7b-v1.3",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "NULL",
        }
    ],
    messages=[{"role": "user", "content": "Hi"}]
)
logger.debug(response)

"""
## For Further Reading

* [Documentation](/docs/Getting-Started) about `autogen`.
* [Documentation](https://github.com/lm-sys/FastChat) about FastChat.
"""
logger.info("## For Further Reading")

logger.info("\n\n[DONE]", bright=True)