from jet.logger import logger
from langchain_experimental.llms import RELLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import logging
import os
import regex  # Note this is the regex library NOT python's re stdlib module
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
# RELLM

[RELLM](https://github.com/r2d4/rellm) is a library that wraps local Hugging Face pipeline models for structured decoding.

It works by generating tokens one at a time. At each step, it masks tokens that don't conform to the provided partial regular expression.


**Warning - this module is still experimental**
"""
logger.info("# RELLM")

# %pip install --upgrade --quiet  rellm langchain-huggingface > /dev/null

"""
### Hugging Face Baseline

First, let's establish a qualitative baseline by checking the output of the model without structured decoding.
"""
logger.info("### Hugging Face Baseline")


logging.basicConfig(level=logging.ERROR)
prompt = """Human: "What's the capital of the United States?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "The capital of the United States is Washington D.C."
}
Human: "What's the capital of Pennsylvania?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "The capital of Pennsylvania is Harrisburg."
}
Human: "What 2 + 5?"
AI Assistant:{
  "action": "Final Answer",
  "action_input": "2 + 5 = 7."
}
Human: 'What's the capital of Maryland?'
AI Assistant:"""


hf_model = pipeline(
    "text-generation", model="cerebras/Cerebras-GPT-590M", max_new_tokens=200
)

original_model = HuggingFacePipeline(pipeline=hf_model)

generated = original_model.generate([prompt], stop=["Human:"])
logger.debug(generated)

"""
***That's not so impressive, is it? It didn't answer the question and it didn't follow the JSON format at all! Let's try with the structured decoder.***

## RELLM LLM Wrapper

Let's try that again, now providing a regex to match the JSON structured format.
"""
logger.info("## RELLM LLM Wrapper")


pattern = regex.compile(
    r'\{\s*"action":\s*"Final Answer",\s*"action_input":\s*(\{.*\}|"[^"]*")\s*\}\nHuman:'
)


model = RELLM(pipeline=hf_model, regex=pattern, max_new_tokens=200)

generated = model.predict(prompt, stop=["Human:"])
logger.debug(generated)

"""
**Voila! Free of parsing errors.**
"""


logger.info("\n\n[DONE]", bright=True)