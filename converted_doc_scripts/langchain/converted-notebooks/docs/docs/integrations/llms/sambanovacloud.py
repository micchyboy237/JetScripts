from jet.logger import logger
from langchain_community.llms.sambanova import SambaNovaCloud
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
# SambaNovaCloud

**[SambaNova](https://sambanova.ai/)'s [SambaNova Cloud](https://cloud.sambanova.ai/)** is a platform for performing inference with open-source models

:::caution
You are currently on a page documenting the use of SambaNovaCloud models as [text completion models](/docs/concepts/text_llms/). We recommend you to use the [chat completion models](/docs/concepts/chat_models).

You may be looking for [SambaNovaCloud Chat Models](/docs/integrations/chat/sambanova/) .
:::

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [SambaNovaCloud](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaNovaCloud.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

This example goes over how to use LangChain to interact with SambaNovaCloud models

## Setup

### Credentials
To access ChatSambaNovaCloud models you will need to create a [SambaNovaCloud account](https://cloud.sambanova.ai/), get an API key and set it as the `SAMBANOVA_API_KEY` environment variable:
"""
logger.info("# SambaNovaCloud")

# import getpass

if "SAMBANOVA_API_KEY" not in os.environ:
#     os.environ["SAMBANOVA_API_KEY"] = getpass.getpass()

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


llm = SambaNovaCloud(
    model="Meta-Llama-3.3-70B-Instruct",
    max_tokens_to_generate=1000,
    temperature=0.01,
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
We can chain our completion model with a prompt template like so:
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

For detailed documentation of all `SambaNovaCloud` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/community/llms/langchain_community.llms.sambanova.SambaNovaCloud.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)