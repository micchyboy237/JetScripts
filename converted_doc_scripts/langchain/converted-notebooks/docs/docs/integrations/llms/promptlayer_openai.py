from jet.logger import logger
from langchain_community.llms import PromptLayerOllama
import os
import promptlayer
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
# PromptLayer Ollama

`PromptLayer` is the first platform that allows you to track, manage, and share your GPT prompt engineering. `PromptLayer` acts a middleware between your code and `Ollama’s` python library.

`PromptLayer` records all your `Ollama API` requests, allowing you to search and explore request history in the `PromptLayer` dashboard.


This example showcases how to connect to [PromptLayer](https://www.promptlayer.com) to start recording your Ollama requests.

Another example is [here](/docs/integrations/providers/promptlayer).

## Install PromptLayer
The `promptlayer` package is required to use PromptLayer with Ollama. Install `promptlayer` using pip.
"""
logger.info("# PromptLayer Ollama")

# %pip install --upgrade --quiet  promptlayer

"""
## Imports
"""
logger.info("## Imports")



"""
## Set the Environment API Key
You can create a PromptLayer API Key at [www.promptlayer.com](https://www.promptlayer.com) by clicking the settings cog in the navbar.

Set it as an environment variable called `PROMPTLAYER_API_KEY`.

# You also need an Ollama Key, called `OPENAI_API_KEY`.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

# PROMPTLAYER_API_KEY = getpass()

os.environ["PROMPTLAYER_API_KEY"] = PROMPTLAYER_API_KEY

# from getpass import getpass

# OPENAI_API_KEY = getpass()

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
## Use the PromptLayerOllama LLM like normal
*You can optionally pass in `pl_tags` to track your requests with PromptLayer's tagging feature.*
"""
logger.info("## Use the PromptLayerOllama LLM like normal")

llm = PromptLayerOllama(pl_tags=["langchain"])
llm("I am a cat and I want")

"""
**The above request should now appear on your [PromptLayer dashboard](https://www.promptlayer.com).**

## Using PromptLayer Track
If you would like to use any of the [PromptLayer tracking features](https://magniv.notion.site/Track-4deee1b1f7a34c1680d085f82567dab9), you need to pass the argument `return_pl_id` when instantiating the PromptLayer LLM to get the request id.
"""
logger.info("## Using PromptLayer Track")

llm = PromptLayerOllama(return_pl_id=True)
llm_results = llm.generate(["Tell me a joke"])

for res in llm_results.generations:
    pl_request_id = res[0].generation_info["pl_request_id"]
    promptlayer.track.score(request_id=pl_request_id, score=100)

"""
Using this allows you to track the performance of your model in the PromptLayer dashboard. If you are using a prompt template, you can attach a template to a request as well.
Overall, this gives you the opportunity to track the performance of different templates and models in the PromptLayer dashboard.
"""
logger.info("Using this allows you to track the performance of your model in the PromptLayer dashboard. If you are using a prompt template, you can attach a template to a request as well.")

logger.info("\n\n[DONE]", bright=True)