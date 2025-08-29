from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import set_global_tokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/llama_2_llama_cpp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LlamaCPP 

In this short notebook, we show how to use the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library with LlamaIndex.

In this notebook, we use the [`Qwen/Qwen2.5-7B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) model, along with the proper prompt formatting. 

By default, if model_path and model_url are blank, the `LlamaCPP` module will load llama2-chat-13B.

## Installation

To get the best performance out of `LlamaCPP`, it is recommended to install the package so that it is compiled with GPU support. A full guide for installing this way is [here](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal).

Full MACOS instructions are also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/).

In general:
- Use `CuBLAS` if you have CUDA and an NVidia GPU
- Use `METAL` if you are running on an M1/M2 MacBook
- Use `CLBLAST` if you are running on an AMD/Intel GPU

For me, on a MAC, I need to install the `metal` backend.

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

Then you can install the required llama-index pacakages
"""
logger.info("# LlamaCPP")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-llama-cpp

"""
## Setup LLM

The LlamaCPP llm is highly configurable. Depending on the model being used, you'll want to pass in `messages_to_prompt` and `completion_to_prompt` functions to help format the model inputs.

For any kwargs that need to be passed in during initialization, set them in `model_kwargs`. A full list of available model kwargs is available in the [LlamaCPP docs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama.Llama.__init__).

For any kwargs that need to be passed in during inference, you can set them in `generate_kwargs`. See the full list of [generate kwargs here](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama.Llama.__call__).

In general, the defaults are a great starting point. The example below shows configuration with all defaults.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("## Setup LLM")

model_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def messages_to_prompt(messages):
    messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def completion_to_prompt(completion):
    messages = [{"role": "user", "content": completion}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=16384,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

"""
We can tell that the model is using `metal` and our GPU due to the logging!

```

```
offloaded 29/29 layers to GPU
```

## Start using our `LlamaCPP` LLM abstraction!

We can simply use the `complete` method of our `LlamaCPP` LLM abstraction to generate completions given a prompt.
"""
logger.info("## Start using our `LlamaCPP` LLM abstraction!")

response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
logger.debug(response.text)

"""
We can use the `stream_complete` endpoint to stream the response as it’s being generated rather than waiting for the entire response to be generated.
"""
logger.info("We can use the `stream_complete` endpoint to stream the response as it’s being generated rather than waiting for the entire response to be generated.")

response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
for response in response_iter:
    logger.debug(response.delta, end="", flush=True)

"""
## Query engine set up with LlamaCPP

We can simply pass in the `LlamaCPP` LLM abstraction to the `LlamaIndex` query engine as usual.

But first, let's change the global tokenizer to match our LLM.
"""
logger.info("## Query engine set up with LlamaCPP")


set_global_tokenizer(
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").encode
)


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)


documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What did the author do growing up?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)