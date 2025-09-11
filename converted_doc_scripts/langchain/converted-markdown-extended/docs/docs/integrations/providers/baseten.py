from jet.logger import logger
from langchain_community.llms import Baseten
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
# Baseten

>[Baseten](https://baseten.co) is a provider of all the infrastructure you need to deploy and serve
> ML models performantly, scalably, and cost-efficiently.

>As a model inference platform, `Baseten` is a `Provider` in the LangChain ecosystem.
The `Baseten` integration currently implements a single `Component`, LLMs, but more are planned!

>`Baseten` lets you run both open source models like Llama 2 or Mistral and run proprietary or
fine-tuned models on dedicated GPUs. If you're used to a provider like Ollama, using Baseten has a few differences:

>* Rather than paying per token, you pay per minute of GPU used.
>* Every model on Baseten uses [Truss](https://truss.baseten.co/welcome), our open-source model packaging framework, for maximum customizability.
>* While we have some [Ollama ChatCompletions-compatible models](https://docs.baseten.co/api-reference/ollama), you can define your own I/O spec with `Truss`.

>[Learn more](https://docs.baseten.co/deploy/lifecycle) about model IDs and deployments.

>Learn more about Baseten in [the Baseten docs](https://docs.baseten.co/).

## Installation and Setup

You'll need two things to use Baseten models with LangChain:

* A [Baseten account](https://baseten.co)
* An [API key](https://docs.baseten.co/observability/api-keys)

Export your API key to your as an environment variable called `BASETEN_API_KEY`.
"""
logger.info("# Baseten")

export BASETEN_API_KEY="paste_your_api_key_here"

"""
## LLMs

See a [usage example](/docs/integrations/llms/baseten).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)