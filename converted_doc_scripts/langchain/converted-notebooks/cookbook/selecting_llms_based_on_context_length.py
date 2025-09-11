from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
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
# Selecting LLMs based on Context Length

Different LLMs have different context lengths. As a very immediate an practical example, Ollama has two versions of GPT-3.5-Turbo: one with 4k context, another with 16k context. This notebook shows how to route between them based on input.
"""
logger.info("# Selecting LLMs based on Context Length")


short_context_model = ChatOllama(model="llama3.2")
long_context_model = ChatOllama(model="llama3.2")

def get_context_length(prompt: PromptValue):
    messages = prompt.to_messages()
    tokens = short_context_model.get_num_tokens_from_messages(messages)
    return tokens

prompt = PromptTemplate.from_template("Summarize this passage: {context}")

def choose_model(prompt: PromptValue):
    context_len = get_context_length(prompt)
    if context_len < 30:
        logger.debug("short model")
        return short_context_model
    else:
        logger.debug("long model")
        return long_context_model

chain = prompt | choose_model | StrOutputParser()

chain.invoke({"context": "a frog went to a pond"})

chain.invoke(
    {"context": "a frog went to a pond and sat on a log and went to a different pond"}
)

logger.info("\n\n[DONE]", bright=True)