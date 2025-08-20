from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.openvino import OpenVINOLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openvino.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OpenVINO LLMs

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. OpenVINO™ Runtime can enable running the same model optimized across various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix). Accelerate your deep learning performance across use cases like: language + LLMs, computer vision, automatic speech recognition, and more.

OpenVINO models can be run locally through `OpenVINOLLM` entitiy wrapped by LlamaIndex :

In the below line, we install the packages necessary for this demo:
"""
logger.info("# OpenVINO LLMs")

# %pip install llama-index-llms-openvino transformers huggingface_hub

"""
Now that we're set up, let's play around:

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("Now that we're set up, let's play around:")

# !pip install llama-index


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"

    return prompt


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

"""
### Model Loading

Models can be loaded by specifying the model parameters using the `OpenVINOLLM` method.

If you have an Intel GPU, you can specify `device_map="gpu"` to run inference on it.
"""
logger.info("### Model Loading")

ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

ov_llm = OpenVINOLLM(
    model_id_or_path="HuggingFaceH4/zephyr-7b-beta",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"ov_config": ov_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="cpu",
)

response = ov_llm.complete("What is the meaning of life?")
logger.debug(str(response))

"""
### Inference with local OpenVINO model

It is possible to [export your model](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export) to the OpenVINO IR format with the CLI, and load the model from local folder.
"""
logger.info("### Inference with local OpenVINO model")

# !optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta ov_model_dir

"""
It is recommended to apply 8 or 4-bit weight quantization to reduce inference latency and model footprint using `--weight-format`:
"""
logger.info("It is recommended to apply 8 or 4-bit weight quantization to reduce inference latency and model footprint using `--weight-format`:")

# !optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta --weight-format int8 ov_model_dir

# !optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta --weight-format int4 ov_model_dir

ov_llm = OpenVINOLLM(
    model_id_or_path="ov_model_dir",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"ov_config": ov_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="gpu",
)

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
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = ov_llm.stream_complete("Who is Paul Graham?")
for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = ov_llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG example with LlamaIndex](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-llamaindex).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)