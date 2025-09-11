from jet.logger import logger
from langchain_community.chat_models import ChatKonko
from langchain_community.llms import Konko
from langchain_core.messages import HumanMessage
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
# Konko
All functionality related to Konko

>[Konko AI](https://www.konko.ai/) provides a fully managed API to help application developers

>1. **Select** the right open source or proprietary LLMs for their application
>2. **Build** applications faster with integrations to leading application frameworks and fully managed APIs
>3. **Fine tune** smaller open-source LLMs to achieve industry-leading performance at a fraction of the cost
>4. **Deploy production-scale APIs** that meet security, privacy, throughput, and latency SLAs without infrastructure set-up or administration using Konko AI's SOC 2 compliant, multi-cloud infrastructure

## Installation and Setup

1. Sign in to our web app to [create an API key](https://platform.konko.ai/settings/api-keys) to access models via our endpoints for [chat completions](https://docs.konko.ai/reference/post-chat-completions) and [completions](https://docs.konko.ai/reference/post-completions).
2. Enable a Python3.8+ environment
3. Install the SDK
"""
logger.info("# Konko")

pip install konko

"""
# 4. Set API Keys as environment variables(`KONKO_API_KEY`,`OPENAI_API_KEY`)
"""
# logger.info("4. Set API Keys as environment variables(`KONKO_API_KEY`,`OPENAI_API_KEY`)")

export KONKO_API_KEY={your_KONKO_API_KEY_here}
# export OPENAI_API_KEY={your_OPENAI_API_KEY_here} #Optional

"""
Please see [the Konko docs](https://docs.konko.ai/docs/getting-started) for more details.


## LLM

**Explore Available Models:** Start by browsing through the [available models](https://docs.konko.ai/docs/list-of-models) on Konko. Each model caters to different use cases and capabilities.

Another way to find the list of models running on the Konko instance is through this [endpoint](https://docs.konko.ai/reference/get-models).

See a usage [example](/docs/integrations/llms/konko).

### Examples of Endpoint Usage

- **Completion with mistralai/Mistral-7B-v0.1:**

  ```python
  llm = Konko(max_tokens=800, model='mistralai/Mistral-7B-v0.1')
  prompt = "Generate a Product Description for Apple Iphone 15"
  response = llm.invoke(prompt)
  ```

## Chat Models

See a usage [example](/docs/integrations/chat/konko).


- **ChatCompletion with Mistral-7B:**

  ```python
  chat_instance = ChatKonko(max_tokens=10, model = 'mistralai/mistral-7b-instruct-v0.1')
  msg = HumanMessage(content="Hi")
  chat_response = chat_instance([msg])
  ```

For further assistance, contact [support@konko.ai](mailto:support@konko.ai) or join our [Discord](https://discord.gg/TXV2s3z7RZ).
"""
logger.info("## LLM")

logger.info("\n\n[DONE]", bright=True)