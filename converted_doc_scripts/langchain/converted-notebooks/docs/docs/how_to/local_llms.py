from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_community.llms import GPT4All
from langchain_community.llms import LlamaCpp
from langchain_community.llms.llamafile import Llamafile
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
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
# Run models locally

## Use case

The popularity of projects like [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://github.com/ollama/ollama), [GPT4All](https://github.com/nomic-ai/gpt4all), [llamafile](https://github.com/Mozilla-Ocho/llamafile), and others underscore the demand to run LLMs locally (on your own device).

This has at least two important benefits:

1. **Privacy**: Your data is not sent to a third party, and it is not subject to the terms of service of a commercial service
2. **Cost**: There is no inference fee, which is important for token-intensive applications (e.g., [long-running simulations](https://twitter.com/RLanceMartin/status/1691097659262820352?s=20), summarization)

## Overview

Running an LLM locally requires a few things:

1. **Open-source LLM**: An open-source LLM that can be freely modified and shared 
2. **Inference**: Ability to run this LLM on your device w/ acceptable latency

### Open-source LLMs

Users can now gain access to a rapidly growing set of [open-source LLMs](https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-better). 

These LLMs can be assessed across at least two dimensions (see figure):
 
1. **Base model**: What is the base-model and how was it trained?
2. **Fine-tuning approach**: Was the base-model fine-tuned and, if so, what [set of instructions](https://cameronrwolfe.substack.com/p/beyond-llama-the-power-of-open-llms#%C2%A7alpaca-an-instruction-following-llama-model) was used?

![Image description](../../static/img/OSS_LLM_overview.png)

The relative performance of these models can be assessed using several leaderboards, including:

1. [LmSys](https://chat.lmsys.org/?arena)
2. [GPT4All](https://gpt4all.io/index.html)
3. [HuggingFace](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

### Inference

A few frameworks for this have emerged to support inference of open-source LLMs on various devices:

1. [`llama.cpp`](https://github.com/ggerganov/llama.cpp): C++ implementation of llama inference code with [weight optimization / quantization](https://finbarr.ca/how-is-llama-cpp-possible/)
2. [`gpt4all`](https://docs.gpt4all.io/index.html): Optimized C backend for inference
3. [`ollama`](https://github.com/ollama/ollama): Bundles model weights and environment into an app that runs on device and serves the LLM
4. [`llamafile`](https://github.com/Mozilla-Ocho/llamafile): Bundles model weights and everything needed to run the model in a single file, allowing you to run the LLM locally from this file without any additional installation steps

In general, these frameworks will do a few things:

1. **Quantization**: Reduce the memory footprint of the raw model weights
2. **Efficient implementation for inference**: Support inference on consumer hardware (e.g., CPU or laptop GPU)

In particular, see [this excellent post](https://finbarr.ca/how-is-llama-cpp-possible/) on the importance of quantization.

![Image description](../../static/img/llama-memory-weights.png)

With less precision, we radically decrease the memory needed to store the LLM in memory.

In addition, we can see the importance of GPU memory bandwidth [sheet](https://docs.google.com/spreadsheets/d/1OehfHHNSn66BP2h3Bxp2NJTVX97icU0GmCXF6pK23H8/edit#gid=0)!

A Mac M2 Max is 5-6x faster than a M1 for inference due to the larger GPU memory bandwidth.

![Image description](../../static/img/llama_t_put.png)

### Formatting prompts

Some providers have [chat model](/docs/concepts/chat_models) wrappers that takes care of formatting your input prompt for the specific local model you're using. However, if you are prompting local models with a [text-in/text-out LLM](/docs/concepts/text_llms) wrapper, you may need to use a prompt tailored for your specific model.

This can [require the inclusion of special tokens](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). [Here's an example for LLaMA 2](https://smith.langchain.com/hub/rlm/rag-prompt-llama).

## Quickstart

[Ollama](https://ollama.com/) is one way to easily run inference on macOS.
 
The instructions [here](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) provide details, which we summarize:
 
* [Download and run](https://ollama.ai/download) the app
* From command line, fetch a model from this [list of options](https://ollama.com/search): e.g., `ollama pull gpt-oss:20b`
* When the app is running, all models are automatically served on `localhost:11434`
"""
logger.info("# Run models locally")

# %pip install -qU jet.adapters.langchain.chat_ollama


llm = ChatOllama(model="llama3.2")

llm.invoke("The first man on the moon was ...").content

"""
Stream tokens as they are being generated:
"""
logger.info("Stream tokens as they are being generated:")

for chunk in llm.stream("The first man on the moon was ..."):
    logger.debug(chunk, end="|", flush=True)

"""
Ollama also includes a chat model wrapper that handles formatting conversation turns:
"""
logger.info("Ollama also includes a chat model wrapper that handles formatting conversation turns:")


chat_model = ChatOllama(model="llama3.2")

chat_model.invoke("Who was the first man on the moon?")

"""
## Environment

Inference speed is a challenge when running models locally (see above).

To minimize latency, it is desirable to run models locally on GPU, which ships with many consumer laptops [e.g., Apple devices](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/).

And even with GPU, the available GPU memory bandwidth (as noted above) is important.

### Running Apple silicon GPU

`ollama` and [`llamafile`](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#gpu-support) will automatically utilize the GPU on Apple devices.
 
Other frameworks require the user to set up the environment to utilize the Apple GPU.

For example, `llama.cpp` python bindings can be configured to use the GPU via [Metal](https://developer.apple.com/metal/).

Metal is a graphics and compute API created by Apple providing near-direct access to the GPU. 

See the [`llama.cpp`](/docs/integrations/llms/llamacpp) setup [here](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md) to enable this.

In particular, ensure that conda is using the correct virtual environment that you created (`miniforge3`).

e.g., for me:

```shell
conda activate /Users/rlm/miniforge3/envs/llama
```

With the above confirmed, then:

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

## LLMs

There are various ways to gain access to quantized model weights.

1. [`HuggingFace`](https://huggingface.co/TheBloke) - Many quantized model are available for download and can be run with framework such as [`llama.cpp`](https://github.com/ggerganov/llama.cpp). You can also download models in [`llamafile` format](https://huggingface.co/models?other=llamafile) from HuggingFace.
2. [`gpt4all`](https://gpt4all.io/index.html) - The model explorer offers a leaderboard of metrics and associated quantized models available for download 
3. [`ollama`](https://github.com/jmorganca/ollama) - Several models can be accessed directly via `pull`

### Ollama

With [Ollama](https://github.com/ollama/ollama), fetch a model via `ollama pull <model family>:<tag>`.
"""
logger.info("## Environment")

llm = ChatOllama(model="llama3.2")
llm.invoke("The first man on the moon was ... think step by step")

"""
### Llama.cpp

Llama.cpp is compatible with a [broad set of models](https://github.com/ggerganov/llama.cpp).

For example, below we run inference on `llama2-13b` with 4 bit quantization downloaded from [HuggingFace](https://huggingface.co/TheBloke/Llama-2-13B-GGML/tree/main).

As noted above, see the [API reference](https://python.langchain.com/api_reference/langchain/llms/langchain.llms.llamacpp.LlamaCpp.html?highlight=llamacpp#langchain.llms.llamacpp.LlamaCpp) for the full set of parameters. 

From the [llama.cpp API reference docs](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.llamacpp.LlamaCpp.html), a few are worth commenting on:

`n_gpu_layers`: number of layers to be loaded into GPU memory

* Value: 1
* Meaning: Only one layer of the model will be loaded into GPU memory (1 is often sufficient).

`n_batch`: number of tokens the model should process in parallel 

* Value: n_batch
* Meaning: It's recommended to choose a value between 1 and n_ctx (which in this case is set to 2048)

`n_ctx`: Token context window

* Value: 2048
* Meaning: The model will consider a window of 2048 tokens at a time

`f16_kv`: whether the model should use half-precision for the key/value cache

* Value: True
* Meaning: The model will use half-precision, which can be more memory efficient; Metal only supports True.
"""
logger.info("### Llama.cpp")

# %env CMAKE_ARGS="-DLLAMA_METAL=on"
# %env FORCE_CMAKE=1
# %pip install --upgrade --quiet  llama-cpp-python --no-cache-dir


llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

"""
The console log will show the below to indicate Metal was enabled properly from steps above:
```
ggml_metal_init: allocating
ggml_metal_init: using MPS
```
"""
logger.info("The console log will show the below to indicate Metal was enabled properly from steps above:")

llm.invoke("The first man on the moon was ... Let's think step by step")

"""
### GPT4All

We can use model weights downloaded from [GPT4All](/docs/integrations/llms/gpt4all) model explorer.

Similar to what is shown above, we can run inference and use [the API reference](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.gpt4all.GPT4All.html) to set parameters of interest.
"""
logger.info("### GPT4All")

# %pip install gpt4all


llm = GPT4All(
    model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin"
)

llm.invoke("The first man on the moon was ... Let's think step by step")

"""
### llamafile

One of the simplest ways to run an LLM locally is using a [llamafile](https://github.com/Mozilla-Ocho/llamafile). All you need to do is:

1) Download a llamafile from [HuggingFace](https://huggingface.co/models?other=llamafile)
2) Make the file executable
3) Run the file

llamafiles bundle model weights and a [specially-compiled](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#technical-details) version of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) into a single file that can run on most computers without any additional dependencies. They also come with an embedded inference server that provides an [API](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints) for interacting with your model. 

Here's a simple bash script that shows all 3 setup steps:

```bash
# Download a llamafile from HuggingFace
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Make the file executable. On Windows, instead just rename the file to end in ".exe".
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Start the model server. Listens at http://localhost:8080 by default.
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
```

After you run the above setup steps, you can use LangChain to interact with your model:
"""
logger.info("### llamafile")


llm = Llamafile()

llm.invoke("The first man on the moon was ... Let's think step by step.")

"""
## Prompts

Some LLMs will benefit from specific prompts.

For example, LLaMA will use [special tokens](https://twitter.com/RLanceMartin/status/1681879318493003776?s=20).

We can use `ConditionalPromptSelector` to set prompt based on the model type.
"""
logger.info("## Prompts")

llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

"""
Set the associated prompt based upon the model version.
"""
logger.info("Set the associated prompt based upon the model version.")


DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
prompt

chain = prompt | llm
question = "What NFL team won the Super Bowl in the year that Justin Bieber was born?"
chain.invoke({"question": question})

"""
We also can use the LangChain Prompt Hub to fetch and / or store prompts that are model specific.

This will work with your [LangSmith API key](https://docs.smith.langchain.com/).

For example, [here](https://smith.langchain.com/hub/rlm/rag-prompt-llama) is a prompt for RAG with LLaMA-specific tokens.

## Use cases

Given an `llm` created from one of the models above, you can use it for [many use cases](/docs/how_to#use-cases).

For example, you can implement a [RAG application](/docs/tutorials/rag) using the chat models demonstrated here.

In general, use cases for local LLMs can be driven by at least two factors:

* **Privacy**: private data (e.g., journals, etc) that a user does not want to share 
* **Cost**: text preprocessing (extraction/tagging), summarization, and agent simulations are token-use-intensive tasks

In addition, [here](https://blog.langchain.dev/using-langsmith-to-support-fine-tuning-of-open-source-llms/) is an overview on fine-tuning, which can utilize open-source LLMs.


"""
logger.info("## Use cases")

logger.info("\n\n[DONE]", bright=True)