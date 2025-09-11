from jet.logger import logger
from langchain_community.llms import GPT4All
from langchain_core.callbacks import BaseCallbackHandler
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
# GPT4All

[GitHub:nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) an ecosystem of open-source chatbots trained on a massive collections of clean assistant data including code, stories and dialogue.

This example goes over how to use LangChain to interact with `GPT4All` models.
"""
logger.info("# GPT4All")

# %pip install --upgrade --quiet langchain-community gpt4all

"""
### Import GPT4All
"""
logger.info("### Import GPT4All")


"""
### Set Up Question to pass to LLM
"""
logger.info("### Set Up Question to pass to LLM")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

"""
### Specify Model

To run locally, download a compatible ggml-formatted model. 
 
The [gpt4all page](https://gpt4all.io/index.html) has a useful `Model Explorer` section:

* Select a model of interest
* Download using the UI and move the `.bin` to the `local_path` (noted below)

For more info, visit https://github.com/nomic-ai/gpt4all.

---

This integration does not yet support streaming in chunks via the [`.stream()`](https://python.langchain.com/docs/how_to/streaming/) method. The below example uses a callback handler with `streaming=True`:
"""
logger.info("### Specify Model")

local_path = (
    "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # replace with your local file path
)


count = 0


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        global count
        if count < 10:
            logger.debug(f"Token: {token}")
            count += 1


llm = GPT4All(model=local_path, callbacks=[MyCustomHandler()], streaming=True)


chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

res = chain.invoke({"question": question})

logger.info("\n\n[DONE]", bright=True)