from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
from voyage_pack_copy.base import VoyageQueryEnginePack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Llama Packs Example

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llama_hub/llama_packs_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This example shows you how to use a simple Llama Pack with VoyageAI. We show the following:
- How to download a Llama Pack
- How to inspect its modules
- How to run it out of the box
- How to customize it.

You can find all packs on https://llamahub.ai

### Setup Data
"""
logger.info("# Llama Packs Example")

# !wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt


reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

"""
### Download and Initialize Pack

We use `download_llama_pack` to download the pack class, and then we initialize it with documents.

Every pack will have different initialization parameters. You can find more about the initialization parameters for each pack through its [README](https://github.com/logan-markewich/llama-hub/tree/main/llama_hub/llama_packs/voyage_query_engine) (also on LlamaHub).

**NOTE**: You must also specify an output directory. In this case the pack is downloaded to `voyage_pack`. This allows you to customize and make changes to the file, and import it later!
"""
logger.info("### Download and Initialize Pack")


VoyageQueryEnginePack = download_llama_pack(
    "VoyageQueryEnginePack", "./voyage_pack"
)

voyage_pack = VoyageQueryEnginePack(documents)

"""
### Inspect Modules
"""
logger.info("### Inspect Modules")

modules = voyage_pack.get_modules()
display(modules)

llm = modules["llm"]
vector_index = modules["index"]

response = llm.complete("hello world")
logger.debug(str(response))

retriever = vector_index.as_retriever()
results = retriever.retrieve("What did the author do growing up?")
logger.debug(str(results[0].get_content()))

"""
### Run Pack

Every pack has a `run` function that will accomplish a certain task out of the box. Here we will go through the full RAG pipeline with VoyageAI embeddings.
"""
logger.info("### Run Pack")

response = voyage_pack.run(
    "What did the author do growing up?", similarity_top_k=2
)

logger.debug(str(response))

"""
### Try Customizing Pack

A major feature of LlamaPacks is that you can and should inspect and modify the code templates!

In this example we'll show how to customize the template with a different LLM, while keeping Voyage embeddings, and then re-use it. We'll use Anthropic instead.

Let's go into `voyage_pack` and create a copy.

1. For demo purposes we'll copy `voyage_pack` into `voyage_pack_copy`.
2. Go into `voyage_pack_copy/base.py` and look at the `VoyageQueryEnginePack` class definition. This is where all the core logic lives. As you can see the pack class itself is a very light base abstraction. You're free to copy/paste the code as you wish.
3. Go into the line in the `__init__` where it do `llm = OllamaFunctionCalling(model="llama3.2")` and instead change it to `llm = Anthropic()` (which defaults to claude-2).
# 4. Do `from llama_index.llms import Anthropic` and ensure that `ANTHROPIC_API_KEY` is set in your env variable.
5. Now you can use!

In the below sections we'll directly re-import the modified `VoyageQueryEnginePack` and use it.
"""
logger.info("### Try Customizing Pack")


voyage_pack = VoyageQueryEnginePack(documents)

response = voyage_pack.run("What did the author do during his time in RISD?")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
