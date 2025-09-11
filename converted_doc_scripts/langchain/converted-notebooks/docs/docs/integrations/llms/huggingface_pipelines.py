from jet.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
# Hugging Face Local Pipelines

Hugging Face models can be run locally through the `HuggingFacePipeline` class.

The [Hugging Face Model Hub](https://huggingface.co/models) hosts over 120k models, 20k datasets, and 50k demo apps (Spaces), all open source and publicly available, in an online platform where people can easily collaborate and build ML together.

These can be called from LangChain either through this local pipeline wrapper or by calling their hosted inference endpoints through the HuggingFaceHub class.

To use, you should have the ``transformers`` python [package installed](https://pypi.org/project/transformers/), as well as [pytorch](https://pytorch.org/get-started/locally/). You can also install `xformer` for a more memory-efficient attention implementation.
"""
logger.info("# Hugging Face Local Pipelines")

# %pip install --upgrade --quiet transformers

"""
### Model Loading

Models can be loaded by specifying the model parameters using the `from_model_id` method.
"""
logger.info("### Model Loading")


hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)

"""
They can also be loaded by passing in an existing `transformers` pipeline directly
"""
logger.info("They can also be loaded by passing in an existing `transformers` pipeline directly")


model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)

"""
### Create Chain

With the model loaded into memory, you can compose it with a prompt to
form a chain.
"""
logger.info("### Create Chain")


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

logger.debug(chain.invoke({"question": question}))

"""
To get response without prompt, you can bind `skip_prompt=True` with LLM.
"""
logger.info("To get response without prompt, you can bind `skip_prompt=True` with LLM.")

chain = prompt | hf.bind(skip_prompt=True)

question = "What is electroencephalography?"

logger.debug(chain.invoke({"question": question}))

"""
Streaming repsonse.
"""
logger.info("Streaming repsonse.")

for chunk in chain.stream(question):
    logger.debug(chunk, end="", flush=True)

"""
### GPU Inference

When running on a machine with GPU, you can specify the `device=n` parameter to put the model on the specified device.
Defaults to `-1` for CPU inference.

If you have multiple-GPUs and/or the model is too large for a single GPU, you can specify `device_map="auto"`, which requires and uses the [Accelerate](https://huggingface.co/docs/accelerate/index) library to automatically determine how to load the model weights. 

*Note*: both `device` and `device_map` should not be specified together and can lead to unexpected behavior.
"""
logger.info("### GPU Inference")

gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=0,  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 10},
)

gpu_chain = prompt | gpu_llm

question = "What is electroencephalography?"

logger.debug(gpu_chain.invoke({"question": question}))

"""
### Batch GPU Inference

If running on a device with GPU, you can also run inference on the GPU in batch mode.
"""
logger.info("### Batch GPU Inference")

gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    device=0,  # -1 for CPU
    batch_size=2,  # adjust as needed based on GPU map and model size.
    model_kwargs={"temperature": 0, "max_length": 64},
)

gpu_chain = prompt | gpu_llm.bind(stop=["\n\n"])

questions = []
for i in range(4):
    questions.append({"question": f"What is the number {i} in french?"})

answers = gpu_chain.batch(questions)
for answer in answers:
    logger.debug(answer)

"""
### Inference with OpenVINO backend

To deploy a model with OpenVINO, you can specify the `backend="openvino"` parameter to trigger OpenVINO as backend inference framework.

If you have an Intel GPU, you can specify `model_kwargs={"device": "GPU"}` to run inference on it.
"""
logger.info("### Inference with OpenVINO backend")

# %pip install --upgrade-strategy eager "optimum[openvino,nncf]" --quiet

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

ov_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": "CPU", "ov_config": ov_config},
    pipeline_kwargs={"max_new_tokens": 10},
)

ov_chain = prompt | ov_llm

question = "What is electroencephalography?"

logger.debug(ov_chain.invoke({"question": question}))

"""
### Inference with local OpenVINO model

It is possible to [export your model](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export) to the OpenVINO IR format with the CLI, and load the model from local folder.
"""
logger.info("### Inference with local OpenVINO model")

# !optimum-cli export openvino --model gpt2 ov_model_dir

"""
It is recommended to apply 8 or 4-bit weight quantization to reduce inference latency and model footprint using `--weight-format`:
"""
logger.info("It is recommended to apply 8 or 4-bit weight quantization to reduce inference latency and model footprint using `--weight-format`:")

# !optimum-cli export openvino --model gpt2  --weight-format int8 ov_model_dir # for 8-bit quantization

# !optimum-cli export openvino --model gpt2  --weight-format int4 ov_model_dir # for 4-bit quantization

ov_llm = HuggingFacePipeline.from_model_id(
    model_id="ov_model_dir",
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": "CPU", "ov_config": ov_config},
    pipeline_kwargs={"max_new_tokens": 10},
)

ov_chain = prompt | ov_llm

question = "What is electroencephalography?"

logger.debug(ov_chain.invoke({"question": question}))

"""
You can get additional inference speed improvement with Dynamic Quantization of activations and KV-cache quantization. These options can be enabled with `ov_config` as follows:
"""
logger.info("You can get additional inference speed improvement with Dynamic Quantization of activations and KV-cache quantization. These options can be enabled with `ov_config` as follows:")

ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

"""
For more information refer to [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html) and [OpenVINO Local Pipelines notebook](/docs/integrations/llms/openvino/).
"""
logger.info("For more information refer to [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html) and [OpenVINO Local Pipelines notebook](/docs/integrations/llms/openvino/).")

logger.info("\n\n[DONE]", bright=True)