from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from ollama import RateLimitError
from unittest.mock import patch
import httpx
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
---
keywords: [LCEL, fallbacks]
---

# How to add fallbacks to a runnable

When working with language models, you may often encounter issues from the underlying APIs, whether these be rate limiting or downtime. Therefore, as you go to move your LLM applications into production it becomes more and more important to safeguard against these. That's why we've introduced the concept of fallbacks. 

A **fallback** is an alternative plan that may be used in an emergency.

Crucially, fallbacks can be applied not only on the LLM level but on the whole runnable level. This is important because often times different models require different prompts. So if your call to Ollama fails, you don't just want to send the same prompt to Ollama - you probably want to use a different prompt template and send a different version there.

## Fallback for LLM API Errors

This is maybe the most common use case for fallbacks. A request to an LLM API can fail for a variety of reasons - the API could be down, you could have hit rate limits, any number of things. Therefore, using fallbacks can help protect against these types of things.

IMPORTANT: By default, a lot of the LLM wrappers catch errors and retry. You will most likely want to turn those off when working with fallbacks. Otherwise the first wrapper will keep on retrying and not failing.
"""
logger.info("# How to add fallbacks to a runnable")

# %pip install --upgrade --quiet  langchain langchain-ollama


"""
First, let's mock out what happens if we hit a RateLimitError from Ollama
"""
logger.info(
    "First, let's mock out what happens if we hit a RateLimitError from Ollama")


request = httpx.Request("GET", "/")
response = httpx.Response(200, request=request)
error = RateLimitError("rate limit", response=response, body="")

ollama_llm = ChatOllama(model="llama3.2")
anthropic_llm = ChatOllama(model="llama3.2")
llm = ollama_llm.with_fallbacks([anthropic_llm])

with patch("ollama.resources.chat.completions.Completions.create", side_effect=error):
    try:
        logger.debug(ollama_llm.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        logger.debug("Hit error")

with patch("ollama.resources.chat.completions.Completions.create", side_effect=error):
    try:
        logger.debug(llm.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        logger.debug("Hit error")

"""
We can use our "LLM with Fallbacks" as we would a normal LLM.
"""
logger.info("We can use our "LLM with Fallbacks" as we would a normal LLM.")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a nice assistant who always includes a compliment in your response",
        ),
        ("human", "Why did the {animal} cross the road"),
    ]
)
chain = prompt | llm
with patch("ollama.resources.chat.completions.Completions.create", side_effect=error):
    try:
        logger.debug(chain.invoke({"animal": "kangaroo"}))
    except RateLimitError:
        logger.debug("Hit error")

"""
## Fallback for Sequences

We can also create fallbacks for sequences, that are sequences themselves. Here we do that with two different models: ChatOllama and then normal Ollama(which does not use a chat model). Because Ollama is NOT a chat model, you likely want a different prompt.
"""
logger.info("## Fallback for Sequences")


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a nice assistant who always includes a compliment in your response",
        ),
        ("human", "Why did the {animal} cross the road"),
    ]
)
chat_model = ChatOllama(model="llama3.2")
bad_chain = chat_prompt | chat_model | StrOutputParser()


prompt_template = """Instructions: You should always include a compliment in your response.

Question: Why did the {animal} cross the road?"""
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOllama()
good_chain = prompt | llm

chain = bad_chain.with_fallbacks([good_chain])
chain.invoke({"animal": "turtle"})

"""
## Fallback for Long Inputs

One of the big limiting factors of LLMs is their context window. Usually, you can count and track the length of prompts before sending them to an LLM, but in situations where that is hard/complicated, you can fallback to a model with a longer context length.
"""
logger.info("## Fallback for Long Inputs")

short_llm = ChatOllama(model="llama3.2")
long_llm = ChatOllama(model="llama3.2")
llm = short_llm.with_fallbacks([long_llm])

inputs = "What is the next number: " + ", ".join(["one", "two"] * 3000)

try:
    logger.debug(short_llm.invoke(inputs))
except Exception as e:
    logger.debug(e)

try:
    logger.debug(llm.invoke(inputs))
except Exception as e:
    logger.debug(e)

"""
## Fallback to Better Model

Often times we ask models to output format in a specific format (like JSON). Models like GPT-3.5 can do this okay, but sometimes struggle. This naturally points to fallbacks - we can try with GPT-3.5 (faster, cheaper), but then if parsing fails we can use GPT-4.
"""
logger.info("## Fallback to Better Model")


prompt = ChatPromptTemplate.from_template(
    "what time was {event} (in %Y-%m-%dT%H:%M:%S.%fZ format - only return this value)"
)

ollama_35 = ChatOllama(model="llama3.2") | DatetimeOutputParser()
ollama_4 = ChatOllama(model="llama3.2") | DatetimeOutputParser()

only_35 = prompt | ollama_35
fallback_4 = prompt | ollama_35.with_fallbacks([ollama_4])

try:
    logger.debug(only_35.invoke({"event": "the superbowl in 1994"}))
except Exception as e:
    logger.debug(f"Error: {e}")

try:
    logger.debug(fallback_4.invoke({"event": "the superbowl in 1994"}))
except Exception as e:
    logger.debug(f"Error: {e}")

logger.info("\n\n[DONE]", bright=True)
