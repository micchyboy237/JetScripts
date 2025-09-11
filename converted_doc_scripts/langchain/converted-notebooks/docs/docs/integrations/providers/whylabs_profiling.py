from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.callbacks import WhyLabsCallbackHandler
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
# WhyLabs

>[WhyLabs](https://docs.whylabs.ai/docs/) is an observability platform designed to monitor data pipelines and ML applications for data quality regressions, data drift, and model performance degradation. Built on top of an open-source package called `whylogs`, the platform enables Data Scientists and Engineers to:
>- Set up in minutes: Begin generating statistical profiles of any dataset using whylogs, the lightweight open-source library.
>- Upload dataset profiles to the WhyLabs platform for centralized and customizable monitoring/alerting of dataset features as well as model inputs, outputs, and performance.
>- Integrate seamlessly: interoperable with any data pipeline, ML infrastructure, or framework. Generate real-time insights into your existing data flow. See more about our integrations here.
>- Scale to terabytes: handle your large-scale data, keeping compute requirements low. Integrate with either batch or streaming data pipelines.
>- Maintain data privacy: WhyLabs relies statistical profiles created via whylogs so your actual data never leaves your environment!
Enable observability to detect inputs and LLM issues faster, deliver continuous improvements, and avoid costly incidents.

## Installation and Setup
"""
logger.info("# WhyLabs")

# %pip install --upgrade --quiet  langkit langchain-ollama langchain

"""
Make sure to set the required API keys and config required to send telemetry to WhyLabs:

* WhyLabs API Key: https://whylabs.ai/whylabs-free-sign-up
* Org and Dataset [https://docs.whylabs.ai/docs/whylabs-onboarding](https://docs.whylabs.ai/docs/whylabs-onboarding#upload-a-profile-to-a-whylabs-project)
* Ollama: https://platform.ollama.com/account/api-keys

Then you can set them like this:

```python

# os.environ["OPENAI_API_KEY"] = ""
os.environ["WHYLABS_DEFAULT_ORG_ID"] = ""
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = ""
os.environ["WHYLABS_API_KEY"] = ""
```
> *Note*: the callback supports directly passing in these variables to the callback, when no auth is directly passed in it will default to the environment. Passing in auth directly allows for writing profiles to multiple projects or organizations in WhyLabs.

## Callbacks

Here's a single LLM integration with Ollama, which will log various out of the box metrics and send telemetry to WhyLabs for monitoring.
"""
logger.info("## Callbacks")


whylabs = WhyLabsCallbackHandler.from_params()
llm = ChatOllama(temperature=0, callbacks=[whylabs])

result = llm.generate(["Hello, World!"])
logger.debug(result)

result = llm.generate(
    [
        "Can you give me 3 SSNs so I can understand the format?",
        "Can you give me 3 fake email addresses?",
        "Can you give me 3 fake US mailing addresses?",
    ]
)
logger.debug(result)
whylabs.close()

logger.info("\n\n[DONE]", bright=True)
