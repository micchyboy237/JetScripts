from jet.logger import logger
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Baseten
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
# Baseten

[Baseten](https://baseten.co) is a [Provider](/docs/integrations/providers/baseten) in the LangChain ecosystem that implements the LLMs component.

This example demonstrates using an LLM — Mistral 7B hosted on Baseten — with LangChain.

# Setup

To run this example, you'll need:

* A [Baseten account](https://baseten.co)
* An [API key](https://docs.baseten.co/observability/api-keys)

Export your API key to your as an environment variable called `BASETEN_API_KEY`.

```sh
export BASETEN_API_KEY="paste_your_api_key_here"
```

# Single model call

First, you'll need to deploy a model to Baseten.

You can deploy foundation models like Mistral and Llama 2 with one click from the [Baseten model library](https://app.baseten.co/explore/) or if you have your own model, [deploy it with Truss](https://truss.baseten.co/welcome).

In this example, we'll work with Mistral 7B. [Deploy Mistral 7B here](https://app.baseten.co/explore/mistral_7b_instruct) and follow along with the deployed model's ID, found in the model dashboard.
"""
logger.info("# Baseten")

# %pip install -qU langchain-community


mistral = Baseten(model="MODEL_ID", deployment="production")

mistral("What is the Mistral wind?")

"""
# Chained model calls

We can chain together multiple calls to one or multiple models, which is the whole point of Langchain!

For example, we can replace GPT with Mistral in this demo of terminal emulation.
"""
logger.info("# Chained model calls")


template = """Assistant is a large language model trained by Ollama.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)


chatgpt_chain = LLMChain(
    llm=mistral,
    llm_kwargs={"max_length": 4096},
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

output = chatgpt_chain.predict(
    human_input="I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is pwd."
)
logger.debug(output)

output = chatgpt_chain.predict(human_input="ls ~")
logger.debug(output)

output = chatgpt_chain.predict(human_input="cd ~")
logger.debug(output)

output = chatgpt_chain.predict(
    human_input="""echo -e "x=lambda y:y*5+3;logger.debug('Result:' + str(x(6)))" > run.py && python3 run.py"""
)
logger.debug(output)

"""
As we can see from the final example, which outputs a number that may or may not be correct, the model is only approximating likely terminal output, not actually executing provided commands. Still, the example demonstrates Mistral's ample context window, code generation capabilities, and ability to stay on-topic even in conversational sequences.
"""
logger.info("As we can see from the final example, which outputs a number that may or may not be correct, the model is only approximating likely terminal output, not actually executing provided commands. Still, the example demonstrates Mistral's ample context window, code generation capabilities, and ability to stay on-topic even in conversational sequences.")

logger.info("\n\n[DONE]", bright=True)