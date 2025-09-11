from jet.logger import logger
from langchain_experimental.llms import LMFormatEnforcer
from langchain_experimental.pydantic_v1 import BaseModel
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import logging
import os
import shutil
import torch


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
# LM Format Enforcer

[LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) is a library that enforces the output format of language models by filtering tokens.

It works by combining a character level parser with a tokenizer prefix tree to allow only the tokens which contains sequences of characters that lead to a potentially valid format.

It supports batched generation.

**Warning - this module is still experimental**
"""
logger.info("# LM Format Enforcer")

# %pip install --upgrade --quiet  lm-format-enforcer langchain-huggingface > /dev/null

"""
### Setting up the model

We will start by setting up a LLama2 model and initializing our desired output format.
Note that Llama2 [requires approval for access to the models](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
"""
logger.info("### Setting up the model")



logging.basicConfig(level=logging.ERROR)


class PlayerInformation(BaseModel):
    first_name: str
    last_name: str
    num_seasons_in_nba: int
    year_of_birth: int


model_id = "meta-llama/Llama-2-7b-chat-hf"

device = "cuda"

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
    )
else:
    raise Exception("GPU not available")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

"""
### HuggingFace Baseline

First, let's establish a qualitative baseline by checking the output of the model without structured decoding.
"""
logger.info("### HuggingFace Baseline")

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

prompt = """Please give me information about {player_name}. You must respond using JSON format, according to the following schema:

{arg_schema}

"""


def make_instruction_prompt(message):
    return f"[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>> {message} [/INST]"


def get_prompt(player_name):
    return make_instruction_prompt(
        prompt.format(
            player_name=player_name, arg_schema=PlayerInformation.schema_json()
        )
    )


hf_model = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
)

original_model = HuggingFacePipeline(pipeline=hf_model)

generated = original_model.predict(get_prompt("Michael Jordan"))
logger.debug(generated)

"""
***The result is usually closer to the JSON object of the schema definition, rather than a json object conforming to the schema. Let's try to enforce proper output.***

## JSONFormer LLM Wrapper

Let's try that again, now providing the Action input's JSON Schema to the model.
"""
logger.info("## JSONFormer LLM Wrapper")


lm_format_enforcer = LMFormatEnforcer(
    json_schema=PlayerInformation.schema(), pipeline=hf_model
)
results = lm_format_enforcer.predict(get_prompt("Michael Jordan"))
logger.debug(results)

"""
**The output conforms to the exact specification! Free of parsing errors.**

This means that if you need to format a JSON for an API call or similar, if you can generate the schema (from a pydantic model or general) you can use this library to make sure that the JSON output is correct, with minimal risk of hallucinations.

### Batch processing

LMFormatEnforcer also works in batch mode:
"""
logger.info("### Batch processing")

prompts = [
    get_prompt(name) for name in ["Michael Jordan", "Kareem Abdul Jabbar", "Tim Duncan"]
]
results = lm_format_enforcer.generate(prompts)
for generation in results.generations:
    logger.debug(generation[0].text)

"""
## Regular Expressions

LMFormatEnforcer has an additional mode, which uses regular expressions to filter the output. Note that it uses [interegular](https://pypi.org/project/interegular/) under the hood, therefore it does not support 100% of the regex capabilities.
"""
logger.info("## Regular Expressions")

question_prompt = "When was Michael Jordan Born? Please answer in mm/dd/yyyy format."
date_regex = r"(0?[1-9]|1[0-2])\/(0?[1-9]|1\d|2\d|3[01])\/(19|20)\d{2}"
answer_regex = " In mm/dd/yyyy format, Michael Jordan was born in " + date_regex

lm_format_enforcer = LMFormatEnforcer(regex=answer_regex, pipeline=hf_model)

full_prompt = make_instruction_prompt(question_prompt)
logger.debug("Unenforced output:")
logger.debug(original_model.predict(full_prompt))
logger.debug("Enforced Output:")
logger.debug(lm_format_enforcer.predict(full_prompt))

"""
As in the previous example, the output conforms to the regular expression and contains the correct information.
"""
logger.info("As in the previous example, the output conforms to the regular expression and contains the correct information.")

logger.info("\n\n[DONE]", bright=True)