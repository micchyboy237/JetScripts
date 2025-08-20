from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ipex_llm import IpexLLM
import os
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# IPEX-LLM on Intel CPU

> [IPEX-LLM](https://github.com/intel-analytics/ipex-llm/) is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency.

This example goes over how to use LlamaIndex to interact with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm/) for text generation and chat on CPU. 

> **Note**
>
> You could refer to [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-llms-ipex-llm/examples) for full examples of `IpexLLM`. Please note that for running on Intel CPU, please specify `-d 'cpu'` in command argument when running the examples.

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.
"""
logger.info("# IPEX-LLM on Intel CPU")

# %pip install llama-index-llms-ipex-llm

"""
In this example we'll use [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model for demostration. It requires updating `transformers` and `tokenizers` packages.
"""
logger.info("In this example we'll use [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model for demostration. It requires updating `transformers` and `tokenizers` packages.")

# %pip install -U transformers==4.37.0 tokenizers==0.15.2

"""
Before loading the Zephyr model, you'll need to define `completion_to_prompt` and `messages_to_prompt` for formatting prompts. This is essential for preparing inputs that the model can interpret accurately.
"""
logger.info("Before loading the Zephyr model, you'll need to define `completion_to_prompt` and `messages_to_prompt` for formatting prompts. This is essential for preparing inputs that the model can interpret accurately.")

def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


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

"""
## Basic Usage

Load the Zephyr model locally using IpexLLM using `IpexLLM.from_model_id`. It will load the model directly in its Huggingface format and convert it automatically to low-bit format for inference.
"""
logger.info("## Basic Usage")


warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*padding_mask.*"
)


llm = IpexLLM.from_model_id(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    context_window=512,
    max_new_tokens=128,
    generate_kwargs={"do_sample": False},
    completion_to_prompt=completion_to_prompt,
    messages_to_prompt=messages_to_prompt,
)

"""
Now you can proceed to use the loaded model for text completion and interactive chat.

### Text Completion
"""
logger.info("### Text Completion")

completion_response = llm.complete("Once upon a time, ")
logger.debug(completion_response.text)

"""
### Streaming Text Completion
"""
logger.info("### Streaming Text Completion")

response_iter = llm.stream_complete("Once upon a time, there's a little girl")
for response in response_iter:
    logger.debug(response.delta, end="", flush=True)

"""
### Chat
"""
logger.info("### Chat")


message = ChatMessage(role="user", content="Explain Big Bang Theory briefly")
resp = llm.chat([message])
logger.debug(resp)

"""
### Streaming Chat
"""
logger.info("### Streaming Chat")

message = ChatMessage(role="user", content="What is AI?")
resp = llm.stream_chat([message], max_tokens=256)
for r in resp:
    logger.debug(r.delta, end="")

"""
## Save/Load Low-bit Model
Alternatively, you might save the low-bit model to disk once and use `from_model_id_low_bit` instead of `from_model_id` to reload it for later use - even across different machines. It is space-efficient, as the low-bit model demands significantly less disk space than the original model. And `from_model_id_low_bit` is also more efficient than `from_model_id` in terms of speed and memory usage, as it skips the model conversion step.

To save the low-bit model, use `save_low_bit` as follows.
"""
logger.info("## Save/Load Low-bit Model")

saved_lowbit_model_path = (
    "./zephyr-7b-alpha-low-bit"  # path to save low-bit model
)

llm._model.save_low_bit(saved_lowbit_model_path)
del llm

"""
Load the model from saved lowbit model path as follows. 
> Note that the saved path for the low-bit model only includes the model itself but not the tokenizers. If you wish to have everything in one place, you will need to manually download or copy the tokenizer files from the original model's directory to the location where the low-bit model is saved.
"""
logger.info("Load the model from saved lowbit model path as follows.")

llm_lowbit = IpexLLM.from_model_id_low_bit(
    model_name=saved_lowbit_model_path,
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    context_window=512,
    max_new_tokens=64,
    completion_to_prompt=completion_to_prompt,
    generate_kwargs={"do_sample": False},
)

"""
Try stream completion using the loaded low-bit model.
"""
logger.info("Try stream completion using the loaded low-bit model.")

response_iter = llm_lowbit.stream_complete("What is Large Language Model?")
for response in response_iter:
    logger.debug(response.delta, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)