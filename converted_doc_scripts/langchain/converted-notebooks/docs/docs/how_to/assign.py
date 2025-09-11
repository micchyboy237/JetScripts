from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
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
sidebar_position: 6
keywords: [RunnablePassthrough, assign, LCEL]
---

# How to add values to a chain's state

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
- [Chaining runnables](/docs/how_to/sequence/)
- [Calling runnables in parallel](/docs/how_to/parallel/)
- [Custom functions](/docs/how_to/functions/)
- [Passing data through](/docs/how_to/passthrough)

:::

An alternate way of [passing data through](/docs/how_to/passthrough) steps of a chain is to leave the current values of the chain state unchanged while assigning a new value under a given key. The [`RunnablePassthrough.assign()`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html#langchain_core.runnables.passthrough.RunnablePassthrough.assign) static method takes an input value and adds the extra arguments passed to the assign function.

This is useful in the common [LangChain Expression Language](/docs/concepts/lcel) pattern of additively creating a dictionary to use as input to a later step.

Here's an example:
"""
logger.info("# How to add values to a chain's state")

# %pip install --upgrade --quiet langchain langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})

"""
Let's break down what's happening here.

- The input to the chain is `{"num": 1}`. This is passed into a `RunnableParallel`, which invokes the runnables it is passed in parallel with that input.
- The value under the `extra` key is invoked. `RunnablePassthrough.assign()` keeps the original keys in the input dict (`{"num": 1}`), and assigns a new key called `mult`. The value is `lambda x: x["num"] * 3)`, which is `3`. Thus, the result is `{"num": 1, "mult": 3}`.
- `{"num": 1, "mult": 3}` is returned to the `RunnableParallel` call, and is set as the value to the key `extra`.
- At the same time, the `modified` key is called. The result is `2`, since the lambda extracts a key called `"num"` from its input and adds one.

Thus, the result is `{'extra': {'num': 1, 'mult': 3}, 'modified': 2}`.

## Streaming

One convenient feature of this method is that it allows values to pass through as soon as they are available. To show this off, we'll use `RunnablePassthrough.assign()` to immediately return source docs in a retrieval chain:
"""
logger.info("## Streaming")


vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OllamaEmbeddings(model="mxbai-embed-large")
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="llama3.2")

generation_chain = prompt | model | StrOutputParser()

retrieval_chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | RunnablePassthrough.assign(output=generation_chain)

stream = retrieval_chain.stream("where did harrison work?")

for chunk in stream:
    logger.debug(chunk)

"""
We can see that the first chunk contains the original `"question"` since that is immediately available. The second chunk contains `"context"` since the retriever finishes second. Finally, the output from the `generation_chain` streams in chunks as soon as it is available.

## Next steps

Now you've learned how to pass data through your chains to help format the data flowing through your chains.

To learn more, see the other how-to guides on runnables in this section.


"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)