from exllamav2.generator import (
ExLlamaV2Sampler,
)
from huggingface_hub import snapshot_download
from jet.logger import logger
from langchain_community.llms.exllamav2 import ExLlamaV2
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from libs.langchain.langchain.chains.llm import LLMChain
import gc
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
# ExLlamaV2

[ExLlamav2](https://github.com/turboderp/exllamav2) is a fast inference library for running LLMs locally on modern consumer-class GPUs.

It supports inference for GPTQ & EXL2 quantized models, which can be accessed on [Hugging Face](https://huggingface.co/TheBloke).

This notebook goes over how to run `exllamav2` within LangChain.

Additional information: 
[ExLlamav2 examples](https://github.com/turboderp/exllamav2/tree/master/examples)

## Installation

Refer to the official [doc](https://github.com/turboderp/exllamav2)
For this notebook, the requirements are : 
- python 3.11
- langchain 0.1.7
- CUDA: 12.1.0 (see bellow)
- torch==2.1.1+cu121
- exllamav2 (0.0.12+cu121) 

If you want to install the same exllamav2 version :
```shell
pip install https://github.com/turboderp/exllamav2/releases/download/v0.0.12/exllamav2-0.0.12+cu121-cp311-cp311-linux_x86_64.whl
```

if you use conda, the dependencies are : 
```
  - conda-forge::ninja
  - nvidia/label/cuda-12.1.0::cuda
  - conda-forge::ffmpeg
  - conda-forge::gxx=11.4
```

## Usage

You don't need an `API_TOKEN` as you will run the LLM locally.

It is worth understanding which models are suitable to be used on the desired machine.

[TheBloke's](https://huggingface.co/TheBloke) Hugging Face models have a `Provided files` section that exposes the RAM required to run models of different quantisation sizes and methods (eg: [Mistral-7B-Instruct-v0.2-GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)).
"""
logger.info("# ExLlamaV2")




def download_GPTQ_model(model_name: str, models_dir: str = "./models/") -> str:
    """Download the model from hugging face repository.

    Params:
    model_name: str: the model name to download (repository name). Example: "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
    """

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    _model_name = model_name.split("/")
    _model_name = "_".join(_model_name)
    model_path = os.path.join(models_dir, _model_name)
    if _model_name not in os.listdir(models_dir):
        snapshot_download(
            repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False
        )
    else:
        logger.debug(f"{model_name} already exists in the models directory")

    return model_path


settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.05

model_path = download_GPTQ_model("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

callbacks = [StreamingStdOutCallbackHandler()]

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = ExLlamaV2(
    model_path=model_path,
    callbacks=callbacks,
    verbose=True,
    settings=settings,
    streaming=True,
    max_new_tokens=150,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What Football team won the UEFA Champions League in the year the iphone 6s was released?"

output = llm_chain.invoke({"question": question})
logger.debug(output)



torch.cuda.empty_cache()
gc.collect()
# !nvidia-smi

logger.info("\n\n[DONE]", bright=True)