from jet.logger import logger
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import GPT4All
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
# GPT4All

This page covers how to use the `GPT4All` wrapper within LangChain. The tutorial is divided into two parts: installation and setup, followed by usage with an example.

## Installation and Setup

- Install the Python package with `pip install gpt4all`
- Download a [GPT4All model](https://gpt4all.io/index.html) and place it in your desired directory

In this example, we are using `mistral-7b-openorca.Q4_0.gguf`:
"""
logger.info("# GPT4All")

mkdir models
wget https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf -O models/mistral-7b-openorca.Q4_0.gguf

"""
## Usage

### GPT4All

To use the GPT4All wrapper, you need to provide the path to the pre-trained model file and the model's configuration.
"""
logger.info("## Usage")


model = GPT4All(model="./models/mistral-7b-openorca.Q4_0.gguf", n_threads=8)

response = model.invoke("Once upon a time, ")

"""
You can also customize the generation parameters, such as `n_predict`, `temp`, `top_p`, `top_k`, and others.

To stream the model's predictions, add in a CallbackManager.
"""
logger.info("You can also customize the generation parameters, such as `n_predict`, `temp`, `top_p`, `top_k`, and others.")



callbacks = [StreamingStdOutCallbackHandler()]
model = GPT4All(model="./models/mistral-7b-openorca.Q4_0.gguf", n_threads=8)

model.invoke("Once upon a time, ", callbacks=callbacks)

"""
## Model File

You can download model files from the GPT4All client. You can download the client from the [GPT4All](https://gpt4all.io/index.html) website.

For a more detailed walkthrough of this, see [this notebook](/docs/integrations/llms/gpt4all)
"""
logger.info("## Model File")

logger.info("\n\n[DONE]", bright=True)