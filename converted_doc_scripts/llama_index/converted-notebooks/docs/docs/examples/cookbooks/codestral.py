from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.mistralai import MistralAI
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/codestral.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Codestral from MistralAI Cookbook

MistralAI released [codestral-latest](https://mistral.ai/news/codestral/) - a code model.

Codestral is a new code model from mistralai tailored for code generation, fluent in over 80 programming languages. It simplifies coding tasks by completing functions, writing tests, and filling in code snippets, enhancing developer efficiency and reducing errors. Codestral operates through a unified API endpoint, making it a versatile tool for software development.

This cookbook showcases how to use the `codestral-latest` model with llama-index. It guides you through using the Codestral fill-in-the-middle and instruct endpoints.

### Setup LLM
"""
logger.info("# Codestral from MistralAI Cookbook")


os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRAL API KEY>"


llm = MistralAI(model="codestral-latest", temperature=0.1)

"""
### Instruct mode usage

#### Write a function for fibonacci
"""
logger.info("### Instruct mode usage")


messages = [ChatMessage(role="user", content="Write a function for fibonacci")]

response = llm.chat(messages)

logger.debug(response)

"""
#### Write a function to build RAG pipeline using LlamaIndex.

Note: The output is mostly accurate, but it is based on an older LlamaIndex package.
"""
logger.info("#### Write a function to build RAG pipeline using LlamaIndex.")

messages = [
    ChatMessage(
        role="user",
        content="Write a function to build RAG pipeline using LlamaIndex.",
    )
]

response = llm.chat(messages)

logger.debug(response)

"""
### Fill-in-the-middle

This feature allows users to set a starting point with a prompt and an optional ending with a suffix and stop. The Codestral model then generates the intervening code, perfect for tasks requiring specific code generation.

#### Fill the code with start and end of the code.
"""
logger.info("### Fill-in-the-middle")

prompt = "def multiply("
suffix = "return a*b"

response = llm.fill_in_middle(prompt, suffix)

logger.debug(
    f"""
{prompt}
{response.text}
{suffix}
"""
)

"""
#### Fill the code with start, end of the code and stop tokens.
"""
logger.info("#### Fill the code with start, end of the code and stop tokens.")

prompt = "def multiply(a,"
suffix = ""
stop = ["\n\n\n"]

response = llm.fill_in_middle(prompt, suffix, stop)

logger.debug(
    f"""
{prompt}
{response.text}
{suffix}
"""
)

logger.info("\n\n[DONE]", bright=True)