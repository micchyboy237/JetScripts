from jet.logger import logger
from langchain_community.llms import Arcee
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
# Arcee
This notebook demonstrates how to use the `Arcee` class for generating text using Arcee's Domain Adapted Language Models (DALMs).
"""
logger.info("# Arcee")

# %pip install -qU langchain-community

"""
### Setup

Before using Arcee, make sure the Arcee API key is set as `ARCEE_API_KEY` environment variable. You can also pass the api key as a named parameter.
"""
logger.info("### Setup")


arcee = Arcee(
    model="DALM-PubMed",
)

"""
### Additional Configuration

You can also configure Arcee's parameters such as `arcee_api_url`, `arcee_app_url`, and `model_kwargs` as needed.
Setting the `model_kwargs` at the object initialization uses the parameters as default for all the subsequent calls to the generate response.
"""
logger.info("### Additional Configuration")

arcee = Arcee(
    model="DALM-Patent",
    arcee_api_url="https://custom-api.arcee.ai",  # default is https://api.arcee.ai
    arcee_app_url="https://custom-app.arcee.ai",  # default is https://app.arcee.ai
    model_kwargs={
        "size": 5,
        "filters": [
            {
                "field_name": "document",
                "filter_type": "fuzzy_search",
                "value": "Einstein",
            }
        ],
    },
)

"""
### Generating Text

You can generate text from Arcee by providing a prompt. Here's an example:
"""
logger.info("### Generating Text")

prompt = "Can AI-driven music therapy contribute to the rehabilitation of patients with disorders of consciousness?"
response = arcee(prompt)

"""
### Additional parameters

Arcee allows you to apply `filters` and set the `size` (in terms of count) of retrieved document(s) to aid text generation. Filters help narrow down the results. Here's how to use these parameters:
"""
logger.info("### Additional parameters")

filters = [
    {"field_name": "document", "filter_type": "fuzzy_search", "value": "Einstein"},
    {"field_name": "year", "filter_type": "strict_search", "value": "1905"},
]

response = arcee(prompt, size=5, filters=filters)

logger.info("\n\n[DONE]", bright=True)