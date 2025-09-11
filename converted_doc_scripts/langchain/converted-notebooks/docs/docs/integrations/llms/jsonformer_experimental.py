from jet.logger import logger
from langchain_core.tools import tool
from langchain_experimental.llms import JsonFormer
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import json
import logging
import os
import requests
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
# JSONFormer

[JSONFormer](https://github.com/1rgs/jsonformer) is a library that wraps local Hugging Face pipeline models for structured decoding of a subset of the JSON Schema.

It works by filling in the structure tokens and then sampling the content tokens from the model.

**Warning - this module is still experimental**
"""
logger.info("# JSONFormer")

# %pip install --upgrade --quiet  jsonformer > /dev/null

"""
### Hugging Face Baseline

First, let's establish a qualitative baseline by checking the output of the model without structured decoding.
"""
logger.info("### Hugging Face Baseline")


logging.basicConfig(level=logging.ERROR)



HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")


@tool
def ask_star_coder(query: str, temperature: float = 1.0, max_new_tokens: float = 250):
    """Query the BigCode StarCoder model about coding questions."""
    url = "https://api-inference.huggingface.co/models/bigcode/starcoder"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "content-type": "application/json",
    }
    payload = {
        "inputs": f"{query}\n\nAnswer:",
        "temperature": temperature,
        "max_new_tokens": int(max_new_tokens),
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8"))

prompt = """You must respond using JSON format, with a single action and single action input.
You may 'ask_star_coder' for help on coding problems.

{arg_schema}

EXAMPLES
----
Human: "So what's all this about a GIL?"
AI Assistant:{{
  "action": "ask_star_coder",
  "action_input": {{"query": "What is a GIL?", "temperature": 0.0, "max_new_tokens": 100}}"
}}
Observation: "The GIL is python's Global Interpreter Lock"
Human: "Could you please write a calculator program in LISP?"
AI Assistant:{{
  "action": "ask_star_coder",
  "action_input": {{"query": "Write a calculator program in LISP", "temperature": 0.0, "max_new_tokens": 250}}
}}
Observation: "(defun add (x y) (+ x y))\n(defun sub (x y) (- x y ))"
Human: "What's the difference between an SVM and an LLM?"
AI Assistant:{{
  "action": "ask_star_coder",
  "action_input": {{"query": "What's the difference between SGD and an SVM?", "temperature": 1.0, "max_new_tokens": 250}}
}}
Observation: "SGD stands for stochastic gradient descent, while an SVM is a Support Vector Machine."

BEGIN! Answer the Human's question as best as you are able.
------
Human: 'What's the difference between an iterator and an iterable?'
AI Assistant:""".format(arg_schema=ask_star_coder.args)


hf_model = pipeline(
    "text-generation", model="cerebras/Cerebras-GPT-590M", max_new_tokens=200
)

original_model = HuggingFacePipeline(pipeline=hf_model)

generated = original_model.predict(prompt, stop=["Observation:", "Human:"])
logger.debug(generated)

"""
***That's not so impressive, is it? It didn't follow the JSON format at all! Let's try with the structured decoder.***

## JSONFormer LLM Wrapper

Let's try that again, now providing a the Action input's JSON Schema to the model.
"""
logger.info("## JSONFormer LLM Wrapper")

decoder_schema = {
    "title": "Decoding Schema",
    "type": "object",
    "properties": {
        "action": {"type": "string", "default": ask_star_coder.name},
        "action_input": {
            "type": "object",
            "properties": ask_star_coder.args,
        },
    },
}


json_former = JsonFormer(json_schema=decoder_schema, pipeline=hf_model)

results = json_former.predict(prompt, stop=["Observation:", "Human:"])
logger.debug(results)

"""
**Voila! Free of parsing errors.**
"""


logger.info("\n\n[DONE]", bright=True)