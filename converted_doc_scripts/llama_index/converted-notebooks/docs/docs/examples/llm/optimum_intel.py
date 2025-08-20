from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.optimum_intel import OptimumIntelLLM
import os
import shutil


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openvino.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Optimum Intel LLMs optimized with IPEX backend

[Optimum Intel](https://github.com/rbrugaro/optimum-intel) accelerates Hugging Face pipelines on Intel architectures leveraging [Intel Extension for Pytorch, (IPEX)](https://github.com/intel/intel-extension-for-pytorch) optimizations

Optimum Intel models can be run locally through `OptimumIntelLLM` entitiy wrapped by LlamaIndex :

In the below line, we install the packages necessary for this demo:
"""
logger.info("# Optimum Intel LLMs optimized with IPEX backend")

# %pip install llama-index-llms-optimum-intel

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

Models can be loaded by specifying the model parameters using the `OptimumIntelLLM` method.
"""
logger.info("### Model Loading")

oi_llm = OptimumIntelLLM(
    model_name="Intel/neural-chat-7b-v3-3",
    tokenizer_name="Intel/neural-chat-7b-v3-3",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="cpu",
)

response = oi_llm.complete("What is the meaning of life?")
logger.debug(str(response))

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = oi_llm.stream_complete("Who is Mother Teresa?")
for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system",
        content="You are an American chef in a small restaurant in New Orleans",
    ),
    ChatMessage(role="user", content="What is your dish of the day?"),
]
resp = oi_llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)