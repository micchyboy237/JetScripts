from jet.logger import logger
from langchain_community.llms.sambanova import SambaStudio
from langchain_core.prompts import PromptTemplate
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
# SambaStudio

**[SambaNova](https://sambanova.ai/)'s** [Sambastudio](https://sambanova.ai/technology/full-stack-ai-platform) is a platform that allows you to train, run batch inference jobs, and deploy online inference endpoints to run open source models that you fine tuned yourself.

:::caution
You are currently on a page documenting the use of SambaStudio models as [text completion models](/docs/concepts/text_llms). We recommend you to use the [chat completion models](/docs/concepts/chat_models).

You may be looking for [SambaStudio Chat Models](/docs/integrations/chat/sambastudio/) .
:::

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [SambaStudio](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaStudio.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

This example goes over how to use LangChain to interact with SambaStudio models

## Setup

### Credentials
A SambaStudio environment is required to deploy a model. Get more information at [sambanova.ai/products/enterprise-ai-platform-sambanova-suite](https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite)

you'll need to [deploy an endpoint](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) and set the `SAMBASTUDIO_URL` and `SAMBASTUDIO_API_KEY` environment variables:
"""
logger.info("# SambaStudio")

# import getpass

if "SAMBASTUDIO_URL" not in os.environ:
#     os.environ["SAMBASTUDIO_URL"] = getpass.getpass()
if "SAMBASTUDIO_API_KEY" not in os.environ:
#     os.environ["SAMBASTUDIO_API_KEY"] = getpass.getpass()

"""
### Installation

The integration lives in the `langchain-community` package. We also need  to install the [sseclient-py](https://pypi.org/project/sseclient-py/) package this is required to run streaming predictions
"""
logger.info("### Installation")

# %pip install --quiet -U langchain-community sseclient-py

"""
## Instantiation
"""
logger.info("## Instantiation")


llm = SambaStudio(
    model_kwargs={
        "do_sample": True,
        "max_tokens": 1024,
        "temperature": 0.01,
        "process_prompt": True,  # set if using CoE endpoints
        "model": "Meta-Llama-3-70B-Instruct-4096",  # set if using CoE endpoints
    },
)

"""
## Invocation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Invocation")

input_text = "Why should I use open source models?"

completion = llm.invoke(input_text)
completion

for chunk in llm.stream("Why should I use open source models?"):
    logger.debug(chunk, end="", flush=True)

"""
## Chaining
"""
logger.info("## Chaining")


prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
## API reference

For detailed documentation of all `SambaStudio` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaStudio.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)