from embedchain import App
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
title: '🛠️ LangSmith'
description: 'Integrate with Langsmith to debug and monitor your LLM app'
---

Embedchain now supports integration with [LangSmith](https://www.langchain.com/langsmith).

To use LangSmith, you need to do the following steps.

1. Have an account on LangSmith and keep the environment variables in handy
2. Set the environment variables in your app so that embedchain has context about it.
3. Just use embedchain and everything will be logged to LangSmith, so that you can better test and monitor your application.

Let's cover each step in detail.


* First make sure that you have created a LangSmith account and have all the necessary variables handy. LangSmith has a [good documentation](https://docs.smith.langchain.com/) on how to get started with their service.

* Once you have setup the account, we will need the following environment variables
"""
logger.info("title: '🛠️ LangSmith'")

export LANGCHAIN_TRACING_V2=true

export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

export LANGCHAIN_API_KEY=<your-api-key>

export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"

"""
If you are using Python, you can use the following code to set environment variables
"""
logger.info("If you are using Python, you can use the following code to set environment variables")


os.environ['LANGCHAIN_TRACING_V2'] = 'true'

os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

os.environ['LANGCHAIN_API_KEY'] = '<your-api-key>'

os.environ['LANGCHAIN_PROJECT'] = '<your-project>'

"""
* Now create an app using Embedchain and everything will be automatically visible in the LangSmith
"""


app = App()

app.add("https://en.wikipedia.org/wiki/Elon_Musk")

app.query("How many companies did Elon found?")

"""
* Now the entire log for this will be visible in langsmith.

<img src="/images/langsmith.png"/>
"""

logger.info("\n\n[DONE]", bright=True)