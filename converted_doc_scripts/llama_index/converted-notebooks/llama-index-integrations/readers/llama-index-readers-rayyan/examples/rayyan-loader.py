from dotenv import load_dotenv
from jet.logger import CustomLogger
from llama_index import VectorStoreIndex, download_loader
from nr_openai_observability import monitor
from time import time
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Jupyter Notebook to test Rayyan Loader

## Install dependencies

```bash
pip install -r notebook-requirements.txt
```

## Configure OllamaFunctionCallingAdapter with your API key

Make sure you have a file named `.env` in the same directory as this notebook, with the following contents:

```
# OPENAI_API_KEY=<your key here>
OPENAI_ORGANIZATION=<your organization here>
```

The organization is optional, but if you are part of multiple organizations, you can specify which one you want to use. Otherwise, the default organization will be used.

Optionally, to enable NewRelic monitoring, add the following to your `.env` file:

```
NEW_RELIC_APP_NAME=<your app name here>
NEW_RELIC_LICENSE_KEY=<your key here>
```
"""
logger.info("# Jupyter Notebook to test Rayyan Loader")


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env.

logger.debug(f"NewRelic application: {os.getenv('NEW_RELIC_APP_NAME')}")

"""
## Load a Rayyan review into LLama Index

Make sure to have a Rayyan credentials file in `rayyan-creds.json`.
Check the [Rayyan SDK](https://github.com/rayyansys/rayyan-python-sdk) for more details.
"""
logger.info("## Load a Rayyan review into LLama Index")



if os.getenv("NEW_RELIC_APP_NAME") and os.getenv("NEW_RELIC_LICENSE_KEY"):
    monitor.initialization(application_name=os.getenv("NEW_RELIC_APP_NAME"))

RayyanReader = download_loader("RayyanReader")
loader = RayyanReader(credentials_path="rayyan-creds.json")

documents = loader.load_data(review_id=746345)
logger.info("Indexing articles...")
t1 = time()
review_index = VectorStoreIndex.from_documents(documents)
t2 = time()
logger.info(f"Done indexing articles in {t2 - t1:.2f} seconds.")

"""
## Query LLama Index about the review data
"""
logger.info("## Query LLama Index about the review data")

query_engine = review_index.as_query_engine()
prompts = [
    "What are the most used interventions?",
    "What is the most common population?",
    "Are there studies about children?",
    "Do we have any studies about COVID-19?",
    "Are there any multi-center randomized clinical trials?",
]
for idx, prompt in enumerate(prompts):
    logger.debug(f"❓ Query {idx + 1}/{len(prompts)}: {prompt}")
    logger.debug("Waiting for response...")
    response = query_engine.query(prompt)
    logger.debug(f"🤖 {response.response}")
    logger.debug("Relevant articles:")
    for article in response.metadata.values():
        logger.debug(f"- [{article['id']}] {article['title']}")
    logger.debug()

logger.info("\n\n[DONE]", bright=True)