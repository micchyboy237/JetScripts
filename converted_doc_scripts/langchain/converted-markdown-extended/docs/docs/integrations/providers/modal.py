from jet.logger import logger
from langchain_community.llms import Modal
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import modal
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
# Modal

This page covers how to use the Modal ecosystem to run LangChain custom LLMs.
It is broken into two parts:

1. Modal installation and web endpoint deployment
2. Using deployed web endpoint with `LLM` wrapper class.

## Installation and Setup

- Install with `pip install modal`
- Run `modal token new`

## Define your Modal Functions and Webhooks

You must include a prompt. There is a rigid response structure:
"""
logger.info("# Modal")

class Item(BaseModel):
    prompt: str

@stub.function()
@modal.web_endpoint(method="POST")
def get_text(item: Item):
    return {"prompt": run_gpt2.call(item.prompt)}

"""
The following is an example with the GPT2 model:
"""
logger.info("The following is an example with the GPT2 model:")



CACHE_PATH = "/root/model_cache"

class Item(BaseModel):
    prompt: str

stub = modal.Stub(name="example-get-started-with-langchain")

def download_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.save_pretrained(CACHE_PATH)
    model.save_pretrained(CACHE_PATH)

image = modal.Image.debian_slim().pip_install(
    "tokenizers", "transformers", "torch", "accelerate"
).run_function(download_model)

@stub.function(
    gpu="any",
    image=image,
    retries=3,
)
def run_gpt2(text: str):
    tokenizer = GPT2Tokenizer.from_pretrained(CACHE_PATH)
    model = GPT2LMHeadModel.from_pretrained(CACHE_PATH)
    encoded_input = tokenizer(text, return_tensors='pt').input_ids
    output = model.generate(encoded_input, max_length=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@stub.function()
@modal.web_endpoint(method="POST")
def get_text(item: Item):
    return {"prompt": run_gpt2.call(item.prompt)}

"""
### Deploy the web endpoint

Deploy the web endpoint to Modal cloud with the [`modal deploy`](https://modal.com/docs/reference/cli/deploy) CLI command.
Your web endpoint will acquire a persistent URL under the `modal.run` domain.

## LLM wrapper around Modal web endpoint

The  `Modal` LLM wrapper class which will accept your deployed web endpoint's URL.
"""
logger.info("### Deploy the web endpoint")


endpoint_url = "https://ecorp--custom-llm-endpoint.modal.run"  # REPLACE ME with your deployed Modal web endpoint's URL

llm = Modal(endpoint_url=endpoint_url)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)