from jet.logger import logger
from langchain_community.llms import Xinference
from langchain_xinference.chat_models import ChatXinference
from langchain_xinference.llms import Xinference
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
# Xorbits Inference (Xinference)

This page demonstrates how to use [Xinference](https://github.com/xorbitsai/inference)
with LangChain.

`Xinference` is a powerful and versatile library designed to serve LLMs,
speech recognition models, and multimodal models, even on your laptop.
With Xorbits Inference, you can effortlessly deploy and serve your or
state-of-the-art built-in models using just a single command.

## Installation and Setup

Xinference can be installed via pip from PyPI:
"""
logger.info("# Xorbits Inference (Xinference)")

pip install "xinference[all]"

"""
## LLM

Xinference supports various models compatible with GGML, including chatglm, baichuan, whisper,
vicuna, and orca. To view the builtin models, run the command:
"""
logger.info("## LLM")

xinference list --all

"""
### Wrapper for Xinference

You can start a local instance of Xinference by running:
"""
logger.info("### Wrapper for Xinference")

xinference

"""
You can also deploy Xinference in a distributed cluster. To do so, first start an Xinference supervisor
on the server you want to run it:
"""
logger.info("You can also deploy Xinference in a distributed cluster. To do so, first start an Xinference supervisor")

xinference-supervisor -H "${supervisor_host}"

"""
Then, start the Xinference workers on each of the other servers where you want to run them on:
"""
logger.info("Then, start the Xinference workers on each of the other servers where you want to run them on:")

xinference-worker -e "http://${supervisor_host}:9997"

"""
You can also start a local instance of Xinference by running:
"""
logger.info("You can also start a local instance of Xinference by running:")

xinference

"""
Once Xinference is running, an endpoint will be accessible for model management via CLI or
Xinference client.

For local deployment, the endpoint will be http://localhost:9997.


For cluster deployment, the endpoint will be http://$\{supervisor_host\}:9997.


Then, you need to launch a model. You can specify the model names and other attributes
including model_size_in_billions and quantization. You can use command line interface (CLI) to
do it. For example,
"""
logger.info("Once Xinference is running, an endpoint will be accessible for model management via CLI or")

xinference launch -n orca -s 3 -q q4_0

"""
A model uid will be returned.

Example usage:
"""
logger.info("A model uid will be returned.")


llm = Xinference(
    server_url="http://0.0.0.0:9997",
    model_uid = {model_uid} # replace model_uid with the model UID return from launching the model
)

llm(
    prompt="Q: where can we visit in the capital of France? A:",
    generate_config={"max_tokens": 1024, "stream": True},
)

"""
### Usage

For more information and detailed examples, refer to the
[example for xinference LLMs](/docs/integrations/llms/xinference)

### Embeddings

Xinference also supports embedding queries and documents. See
[example for xinference embeddings](/docs/integrations/text_embedding/xinference)
for a more detailed demo.


### Xinference LangChain partner package install
Install the integration package with:
"""
logger.info("### Usage")

pip install langchain-xinference

"""
## Chat Models
"""
logger.info("## Chat Models")


"""
## LLM
"""
logger.info("## LLM")


logger.info("\n\n[DONE]", bright=True)