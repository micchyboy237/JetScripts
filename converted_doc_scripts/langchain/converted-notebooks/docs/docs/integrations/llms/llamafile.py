from jet.logger import logger
from langchain_community.llms.llamafile import Llamafile
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
# Llamafile

[Llamafile](https://github.com/Mozilla-Ocho/llamafile) lets you distribute and run LLMs with a single file.

Llamafile does this by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most computers, with no installation.

## Setup

1. Download a llamafile for the model you'd like to use. You can find many models in llamafile format on [HuggingFace](https://huggingface.co/models?other=llamafile). In this guide, we will download a small one, `TinyLlama-1.1B-Chat-v1.0.Q5_K_M`. Note: if you don't have `wget`, you can just download the model via this [link](https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile?download=true).

```bash
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

2. Make the llamafile executable. First, if you haven't done so already, open a terminal. **If you're using MacOS, Linux, or BSD,** you'll need to grant permission for your computer to execute this new file using `chmod` (see below). **If you're on Windows,** rename the file by adding ".exe" to the end (model file should be named `TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile.exe`).


```bash
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile  # run if you're on MacOS, Linux, or BSD
```

3. Run the llamafile in "server mode":

```bash
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser
```

Now you can make calls to the llamafile's REST API. By default, the llamafile server listens at http://localhost:8080. You can find full server documentation [here](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints). You can interact with the llamafile directly via the REST API, but here we'll show how to interact with it using LangChain.

## Usage
"""
logger.info("# Llamafile")


llm = Llamafile()

llm.invoke("Tell me a joke")

"""
To stream tokens, use the `.stream(...)` method:
"""
logger.info("To stream tokens, use the `.stream(...)` method:")

query = "Tell me a joke"

for chunks in llm.stream(query):
    logger.debug(chunks, end="")

logger.debug()

"""
To learn more about the LangChain Expressive Language and the available methods on an LLM, see the [LCEL Interface](/docs/concepts/runnables)
"""
logger.info("To learn more about the LangChain Expressive Language and the available methods on an LLM, see the [LCEL Interface](/docs/concepts/runnables)")

logger.info("\n\n[DONE]", bright=True)