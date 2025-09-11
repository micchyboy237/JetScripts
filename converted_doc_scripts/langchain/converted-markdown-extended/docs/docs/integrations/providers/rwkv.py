from jet.logger import logger
from langchain_community.llms import RWKV
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
# RWKV-4

This page covers how to use the `RWKV-4` wrapper within LangChain.
It is broken into two parts: installation and setup, and then usage with an example.

## Installation and Setup
- Install the Python package with `pip install rwkv`
- Install the tokenizer Python package with `pip install tokenizer`
- Download a [RWKV model](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main) and place it in your desired directory
- Download the [tokens file](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json)

## Usage

### RWKV

To use the RWKV wrapper, you need to provide the path to the pre-trained model file and the tokenizer's configuration.
"""
logger.info("# RWKV-4")


"""
def generate_prompt(instruction, input=None):
    if input:
        return f"""
logger.info("def generate_prompt(instruction, input=None):")Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""
logger.info("else:")Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""


model = RWKV(model="./models/RWKV-4-Raven-3B-v7-Eng-20230404-ctx4096.pth", strategy="cpu fp32", tokens_path="./rwkv/20B_tokenizer.json")
response = model.invoke(generate_prompt("Once upon a time, "))

## Model File

You can find links to model file downloads at the [RWKV-4-Raven](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main) repository.

### Rwkv-4 models -> recommended VRAM

RWKV VRAM
Model | 8bit | bf16/fp16 | fp32
14B   | 16GB | 28GB      | >50GB
7B    | 8GB  | 14GB      | 28GB
3B    | 2.8GB| 6GB       | 12GB
1b5   | 1.3GB| 3GB       | 6GB
"""
logger.info("## Model File")

logger.info("\n\n[DONE]", bright=True)