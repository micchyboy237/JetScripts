from jet.logger import logger
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import Predibase
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
# Predibase

[Predibase](https://predibase.com/) allows you to train, fine-tune, and deploy any ML model—from linear regression to large language model. 

This example demonstrates using Langchain with models deployed on Predibase

# Setup

To run this notebook, you'll need a [Predibase account](https://predibase.com/free-trial/?utm_source=langchain) and an [API key](https://docs.predibase.com/sdk-guide/intro).

You'll also need to install the Predibase Python package:
"""
logger.info("# Predibase")

# %pip install --upgrade --quiet  predibase

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

"""
## Initial Call
"""
logger.info("## Initial Call")


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
)


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)

response = model.invoke(
    "Can you recommend me a nice dry wine?",
    **{"temperature": 0.5, "max_new_tokens": 1024},
)
logger.debug(response)

"""
## Chain Call Setup
"""
logger.info("## Chain Call Setup")


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)

model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)

llm = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)

"""
##  SequentialChain
"""
logger.info("##  SequentialChain")


template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)


overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True
)

review = overall_chain.run("Tragedy at sunset on the beach")

"""
## Fine-tuned LLM (Use your own fine-tuned LLM from Predibase)
"""
logger.info("## Fine-tuned LLM (Use your own fine-tuned LLM from Predibase)")


model = Predibase(
    model="my-base-LLM",
    predibase_api_key=os.environ.get(
        "PREDIBASE_API_TOKEN"
    ),  # Adapter argument is optional.
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="my-finetuned-adapter-id",  # Supports both, Predibase-hosted and HuggingFace-hosted adapter repositories.
    adapter_version=1,  # required for Predibase-hosted adapters (ignored for HuggingFace-hosted adapters)
    **{
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    },
)

logger.info("\n\n[DONE]", bright=True)