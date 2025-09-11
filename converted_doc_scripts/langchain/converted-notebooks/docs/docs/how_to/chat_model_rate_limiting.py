from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.rate_limiters import InMemoryRateLimiter
import os
import shutil
import time


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
# How to handle rate limits

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [LLMs](/docs/concepts/text_llms)
:::


You may find yourself in a situation where you are getting rate limited by the model provider API because you're making too many requests.

For example, this might happen if you are running many parallel queries to benchmark the chat model on a test dataset.

If you are facing such a situation, you can use a rate limiter to help match the rate at which you're making request to the rate allowed
by the API.

:::info Requires ``langchain-core >= 0.2.24``

This functionality was added in ``langchain-core == 0.2.24``. Please make sure your package is up to date.
:::

## Initialize a rate limiter

Langchain comes with a built-in in memory rate limiter. This rate limiter is thread safe and can be shared by multiple threads in the same process.

The provided rate limiter can only limit the number of requests per unit time. It will not help if you need to also limit based on the size
of the requests.
"""
logger.info("# How to handle rate limits")


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

"""
## Choose a model

Choose any model and pass to it the rate_limiter via the `rate_limiter` attribute.
"""
logger.info("## Choose a model")

# from getpass import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass()



model = ChatOllama(model="llama3.2")

"""
Let's confirm that the rate limiter works. We should only be able to invoke the model once per 10 seconds.
"""
logger.info("Let's confirm that the rate limiter works. We should only be able to invoke the model once per 10 seconds.")

for _ in range(5):
    tic = time.time()
    model.invoke("hello")
    toc = time.time()
    logger.debug(toc - tic)

logger.info("\n\n[DONE]", bright=True)