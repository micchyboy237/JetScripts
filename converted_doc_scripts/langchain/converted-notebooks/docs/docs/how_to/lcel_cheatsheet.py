from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableParallel
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tracers.schemas import Run
from typing import Any, Optional
from typing import Optional
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
# LangChain Expression Language Cheatsheet

This is a quick reference for all the most important [LCEL](/docs/concepts/lcel/) primitives. For more advanced usage see the [LCEL how-to guides](/docs/how_to/#langchain-expression-language-lcel) and the [full API reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html).

### Invoke a runnable
#### [Runnable.invoke()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.invoke) / [Runnable.ainvoke()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.ainvoke)
"""
logger.info("# LangChain Expression Language Cheatsheet")


runnable = RunnableLhambda(lambda x: str(x))
runnable.invoke(5)

"""
### Batch a runnable
#### [Runnable.batch()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.batch) / [Runnable.abatch()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.abatch)
"""
logger.info("### Batch a runnable")


runnable = RunnableLambda(lambda x: str(x))
runnable.batch([7, 8, 9])

"""
### Stream a runnable
#### [Runnable.stream()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) / [Runnable.astream()](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream)
"""
logger.info("### Stream a runnable")


def func(x):
    for y in x:
        yield str(y)


runnable = RunnableLambda(func)

for chunk in runnable.stream(range(5)):
    logger.debug(chunk)

"""
### Compose runnables
#### Pipe operator `|`
"""
logger.info("### Compose runnables")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)

chain = runnable1 | runnable2

chain.invoke(2)

"""
### Invoke runnables in parallel
#### [RunnableParallel](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableParallel.html)
"""
logger.info("### Invoke runnables in parallel")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)

chain = RunnableParallel(first=runnable1, second=runnable2)

chain.invoke(2)

"""
### Turn any function into a runnable
#### [RunnableLambda](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)
"""
logger.info("### Turn any function into a runnable")


def func(x):
    return x + 5


runnable = RunnableLambda(func)
runnable.invoke(2)

"""
### Merge input and output dicts
#### [RunnablePassthrough.assign](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
"""
logger.info("### Merge input and output dicts")


runnable1 = RunnableLambda(lambda x: x["foo"] + 7)

chain = RunnablePassthrough.assign(bar=runnable1)

chain.invoke({"foo": 10})

"""
### Include input dict in output dict
#### [RunnablePassthrough](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
"""
logger.info("### Include input dict in output dict")


runnable1 = RunnableLambda(lambda x: x["foo"] + 7)

chain = RunnableParallel(bar=runnable1, baz=RunnablePassthrough())

chain.invoke({"foo": 10})

"""
### Add default invocation args
#### [Runnable.bind](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.bind)
"""
logger.info("### Add default invocation args")


def func(main_arg: dict, other_arg: Optional[str] = None) -> dict:
    if other_arg:
        return {**main_arg, **{"foo": other_arg}}
    return main_arg


runnable1 = RunnableLambda(func)
bound_runnable1 = runnable1.bind(other_arg="bye")

bound_runnable1.invoke({"bar": "hello"})

"""
### Add fallbacks
#### [Runnable.with_fallbacks](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_fallbacks)
"""
logger.info("### Add fallbacks")


runnable1 = RunnableLambda(lambda x: x + "foo")
runnable2 = RunnableLambda(lambda x: str(x) + "foo")

chain = runnable1.with_fallbacks([runnable2])

chain.invoke(5)

"""
### Add retries
#### [Runnable.with_retry](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_retry)
"""
logger.info("### Add retries")


counter = -1


def func(x):
    global counter
    counter += 1
    logger.debug(f"attempt with {counter=}")
    return x / counter


chain = RunnableLambda(func).with_retry(stop_after_attempt=2)

chain.invoke(2)

"""
### Configure runnable execution
#### [RunnableConfig](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html)
"""
logger.info("### Configure runnable execution")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)
runnable3 = RunnableLambda(lambda x: str(x))

chain = RunnableParallel(first=runnable1, second=runnable2, third=runnable3)

chain.invoke(7, config={"max_concurrency": 2})

"""
### Add default config to runnable
#### [Runnable.with_config](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config)
"""
logger.info("### Add default config to runnable")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)
runnable3 = RunnableLambda(lambda x: str(x))

chain = RunnableParallel(first=runnable1, second=runnable2, third=runnable3)
configured_chain = chain.with_config(max_concurrency=2)

chain.invoke(7)

"""
### Make runnable attributes configurable
#### [Runnable.with_configurable_fields](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSerializable.html#langchain_core.runnables.base.RunnableSerializable.configurable_fields)
"""
logger.info("### Make runnable attributes configurable")


class FooRunnable(RunnableSerializable[dict, dict]):
    output_key: str

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list:
        return self._call_with_config(self.subtract_seven, input, config, **kwargs)

    def subtract_seven(self, input: dict) -> dict:
        return {self.output_key: input["foo"] - 7}


runnable1 = FooRunnable(output_key="bar")
configurable_runnable1 = runnable1.configurable_fields(
    output_key=ConfigurableField(id="output_key")
)

configurable_runnable1.invoke(
    {"foo": 10}, config={"configurable": {"output_key": "not bar"}}
)

configurable_runnable1.invoke({"foo": 10})

"""
### Make chain components configurable
#### [Runnable.with_configurable_alternatives](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSerializable.html#langchain_core.runnables.base.RunnableSerializable.configurable_alternatives)
"""
logger.info("### Make chain components configurable")


class ListRunnable(RunnableSerializable[Any, list]):
    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list:
        return self._call_with_config(self.listify, input, config, **kwargs)

    def listify(self, input: Any) -> list:
        return [input]


class StrRunnable(RunnableSerializable[Any, str]):
    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list:
        return self._call_with_config(self.strify, input, config, **kwargs)

    def strify(self, input: Any) -> str:
        return str(input)


runnable1 = RunnableLambda(lambda x: {"foo": x})

configurable_runnable = ListRunnable().configurable_alternatives(
    ConfigurableField(id="second_step"), default_key="list", string=StrRunnable()
)
chain = runnable1 | configurable_runnable

chain.invoke(7, config={"configurable": {"second_step": "string"}})

chain.invoke(7)

"""
### Build a chain dynamically based on input
"""
logger.info("### Build a chain dynamically based on input")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)

chain = RunnableLambda(lambda x: runnable1 if x > 6 else runnable2)

chain.invoke(7)

chain.invoke(5)

"""
### Generate a stream of events
#### [Runnable.astream_events](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events)
"""
logger.info("### Generate a stream of events")

# import nest_asyncio

# nest_asyncio.apply()


runnable1 = RunnableLambda(lambda x: {"foo": x}, name="first")


async def func(x):
    for _ in range(5):
        yield x


runnable2 = RunnableLambda(func, name="second")

chain = runnable1 | runnable2

for event in chain.stream_events("bar", version="v2"):
    logger.debug(
        f"event={event['event']} | name={event['name']} | data={event['data']}")

"""
### Yield batched outputs as they complete
#### [Runnable.batch_as_completed](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.batch_as_completed) / [Runnable.abatch_as_completed](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.abatch_as_completed)
"""
logger.info("### Yield batched outputs as they complete")


runnable1 = RunnableLambda(lambda x: time.sleep(x)
                           or logger.debug(f"slept {x}"))

for idx, result in runnable1.batch_as_completed([5, 1]):
    logger.debug(idx, result)

"""
### Return subset of output dict
#### [Runnable.pick](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.pick)
"""
logger.info("### Return subset of output dict")


runnable1 = RunnableLambda(lambda x: x["baz"] + 5)
chain = RunnablePassthrough.assign(foo=runnable1).pick(["foo", "bar"])

chain.invoke({"bar": "hi", "baz": 2})

"""
### Declaratively make a batched version of a runnable
#### [Runnable.map](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.map)
"""
logger.info("### Declaratively make a batched version of a runnable")


runnable1 = RunnableLambda(lambda x: list(range(x)))
runnable2 = RunnableLambda(lambda x: x + 5)

chain = runnable1 | runnable2.map()

chain.invoke(3)

"""
### Get a graph representation of a runnable
#### [Runnable.get_graph](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.get_graph)
"""
logger.info("### Get a graph representation of a runnable")


runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)
runnable3 = RunnableLambda(lambda x: str(x))

chain = runnable1 | RunnableParallel(second=runnable2, third=runnable3)

chain.get_graph().print_ascii()

"""
### Get all prompts in a chain
#### [Runnable.get_prompts](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.get_prompts)
"""
logger.info("### Get all prompts in a chain")


prompt1 = ChatPromptTemplate.from_messages(
    [("system", "good ai"), ("human", "{input}")]
)
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "really good ai"),
        ("human", "{input}"),
        ("ai", "{ai_output}"),
        ("human", "{input2}"),
    ]
)
fake_llm = RunnableLambda(lambda prompt: "i am good ai")
chain = prompt1.assign(ai_output=fake_llm) | prompt2 | fake_llm

for i, prompt in enumerate(chain.get_prompts()):
    logger.debug(f"**prompt {i=}**\n")
    logger.debug(prompt.pretty_repr())
    logger.debug("\n" * 3)

"""
### Add lifecycle listeners
#### [Runnable.with_listeners](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_listeners)
"""
logger.info("### Add lifecycle listeners")


def on_start(run_obj: Run):
    logger.debug("start_time:", run_obj.start_time)


def on_end(run_obj: Run):
    logger.debug("end_time:", run_obj.end_time)


runnable1 = RunnableLambda(lambda x: time.sleep(x))
chain = runnable1.with_listeners(on_start=on_start, on_end=on_end)
chain.invoke(2)

logger.info("\n\n[DONE]", bright=True)
