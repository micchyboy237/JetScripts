from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.prompts.prompt import PromptTemplate
from langchain_experimental.llm_bash.base import LLMBashChain
from langchain_experimental.llm_bash.bash import BashProcess
from langchain_experimental.llm_bash.prompt import BashOutputParser
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
# Bash chain
This notebook showcases using LLMs and a bash process to perform simple filesystem commands.
"""
logger.info("# Bash chain")


llm = Ollama(temperature=0)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain = LLMBashChain.from_llm(llm, verbose=True)

bash_chain.invoke(text)

"""
## Customize Prompt
You can also customize the prompt that is used. Here is an example prompting to avoid using the 'echo' utility
"""
logger.info("## Customize Prompt")


_PROMPT_TEMPLATE = """If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:
Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"
I need to take the following actions:
- List all files in the directory
- Create a new directory
- Copy the files from the first directory into the second directory
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

Do not use 'echo' when writing the script.

That is the format. Begin!
Question: {question}"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
    output_parser=BashOutputParser(),
)

bash_chain = LLMBashChain.from_llm(llm, prompt=PROMPT, verbose=True)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain.invoke(text)

"""
## Persistent Terminal

By default, the chain will run in a separate subprocess each time it is called. This behavior can be changed by instantiating with a persistent bash process.
"""
logger.info("## Persistent Terminal")


persistent_process = BashProcess(persistent=True)
bash_chain = LLMBashChain.from_llm(llm, bash_process=persistent_process, verbose=True)

text = "List the current directory then move up a level."

bash_chain.invoke(text)

bash_chain.invoke(text)

logger.info("\n\n[DONE]", bright=True)