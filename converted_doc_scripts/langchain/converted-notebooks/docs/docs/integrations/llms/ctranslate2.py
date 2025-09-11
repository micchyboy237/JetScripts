from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import CTranslate2
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
# CTranslate2

**CTranslate2** is a C++ and Python library for efficient inference with Transformer models.

The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to accelerate and reduce the memory usage of Transformer models on CPU and GPU.

Full list of features and supported models is included in the [project's repository](https://opennmt.net/CTranslate2/guides/transformers.html). To start, please check out the official [quickstart guide](https://opennmt.net/CTranslate2/quickstart.html).

To use, you should have `ctranslate2` python package installed.
"""
logger.info("# CTranslate2")

# %pip install --upgrade --quiet  ctranslate2

"""
To use a Hugging Face model with CTranslate2, it has to be first converted to CTranslate2 format using the `ct2-transformers-converter` command. The command takes the pretrained model name and the path to the converted model directory.
"""
logger.info("To use a Hugging Face model with CTranslate2, it has to be first converted to CTranslate2 format using the `ct2-transformers-converter` command. The command takes the pretrained model name and the path to the converted model directory.")

# !ct2-transformers-converter --model meta-llama/Llama-2-7b-hf --quantization bfloat16 --output_dir ./llama-2-7b-ct2 --force


llm = CTranslate2(
    model_path="./llama-2-7b-ct2",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    device_index=[0, 1],
    compute_type="bfloat16",
)

"""
## Single call
"""
logger.info("## Single call")

logger.debug(
    llm.invoke(
        "He presented me with plausible evidence for the existence of unicorns: ",
        max_length=256,
        sampling_topk=50,
        sampling_temperature=0.2,
        repetition_penalty=2,
        cache_static_prompt=False,
    )
)

"""
## Multiple calls:
"""
logger.info("## Multiple calls:")

logger.debug(
    llm.generate(
        ["The list of top romantic songs:\n1.", "The list of top rap songs:\n1."],
        max_length=128,
    )
)

"""
## Integrate the model in an LLMChain
"""
logger.info("## Integrate the model in an LLMChain")


template = """{question}

Let's think step by step. """
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

logger.debug(llm_chain.run(question))

logger.info("\n\n[DONE]", bright=True)