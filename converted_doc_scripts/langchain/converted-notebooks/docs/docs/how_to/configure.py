from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
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
sidebar_position: 7
keywords: [ConfigurableField, configurable_fields, ConfigurableAlternatives, configurable_alternatives, LCEL]
---

# How to configure runtime chain internals

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [The Runnable interface](/docs/concepts/runnables/)
- [Chaining runnables](/docs/how_to/sequence/)
- [Binding runtime arguments](/docs/how_to/binding/)

:::

Sometimes you may want to experiment with, or even expose to the end user, multiple different ways of doing things within your chains.
This can include tweaking parameters such as temperature or even swapping out one model for another.
In order to make this experience as easy as possible, we have defined two methods.

- A `configurable_fields` method. This lets you configure particular fields of a runnable.
  - This is related to the [`.bind`](/docs/how_to/binding) method on runnables, but allows you to specify parameters for a given step in a chain at runtime rather than specifying them beforehand.
- A `configurable_alternatives` method. With this method, you can list out alternatives for any particular runnable that can be set during runtime, and swap them for those specified alternatives.

## Configurable Fields

Let's walk through an example that configures chat model fields like temperature at runtime:
"""
logger.info("# How to configure runtime chain internals")

# %pip install --upgrade --quiet langchain langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
### Configuring fields on a chat model

If using [init_chat_model](/docs/how_to/chat_models_universal_init/) to create a chat model, you can specify configurable fields in the constructor:
"""
logger.info("### Configuring fields on a chat model")


llm = init_chat_model(
    "ollama:llama3.2",
    configurable_fields=("temperature",),
)

"""
You can then set the parameter at runtime using `.with_config`:
"""
logger.info("You can then set the parameter at runtime using `.with_config`:")

response = llm.with_config({"temperature": 0}).invoke("Hello")
logger.debug(response.content)

"""
:::tip

In addition to invocation parameters like temperature, configuring fields this way extends to clients and other attributes.

:::

#### Use with tools

This method is applicable when [binding tools](/docs/concepts/tool_calling/) as well:
"""
logger.info("#### Use with tools")



@tool
def get_weather(location: str):
    """Get the weather."""
    return "It's sunny."


llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.with_config({"temperature": 0}).invoke(
    "What's the weather in SF?"
)
response.tool_calls

"""
In addition to `.with_config`, we can now include the parameter when passing a configuration directly. See example below, where we allow the underlying model temperature to be configurable inside of a [langgraph agent](/docs/tutorials/agents/):
"""
logger.info("In addition to `.with_config`, we can now include the parameter when passing a configuration directly. See example below, where we allow the underlying model temperature to be configurable inside of a [langgraph agent](/docs/tutorials/agents/):")

# ! pip install --upgrade langgraph


agent = create_react_agent(llm, [get_weather])

response = agent.invoke(
    {"messages": "What's the weather in Boston?"},
    {"configurable": {"temperature": 0}},
)

"""
### Configuring fields on arbitrary Runnables

You can also use the `.configurable_fields` method on arbitrary [Runnables](/docs/concepts/runnables/), as shown below:
"""
logger.info("### Configuring fields on arbitrary Runnables")


model = ChatOllama(model="llama3.2").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

model.invoke("pick a random number")

"""
Above, we defined `temperature` as a [`ConfigurableField`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html#langchain_core.runnables.utils.ConfigurableField) that we can set at runtime. To do so, we use the [`with_config`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config) method like this:
"""
logger.info("Above, we defined `temperature` as a [`ConfigurableField`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.utils.ConfigurableField.html#langchain_core.runnables.utils.ConfigurableField) that we can set at runtime. To do so, we use the [`with_config`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config) method like this:")

model.with_config(configurable={"llm_temperature": 0.9}).invoke("pick a random number")

"""
Note that the passed `llm_temperature` entry in the dict has the same key as the `id` of the `ConfigurableField`.

We can also do this to affect just one step that's part of a chain:
"""
logger.info("Note that the passed `llm_temperature` entry in the dict has the same key as the `id` of the `ConfigurableField`.")

prompt = PromptTemplate.from_template("Pick a random number above {x}")
chain = prompt | model

chain.invoke({"x": 0})

chain.with_config(configurable={"llm_temperature": 0.9}).invoke({"x": 0})

"""
## Configurable Alternatives

The `configurable_alternatives()` method allows us to swap out steps in a chain with an alternative. Below, we swap out one chat model for another:
"""
logger.info("## Configurable Alternatives")

# %pip install --upgrade --quiet langchain-anthropic

# from getpass import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass()


llm = ChatOllama(
    model="claude-3-haiku-20240307", temperature=0
).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="anthropic",
    ollama=ChatOllama(model="llama3.2"),
    gpt4=ChatOllama(model="llama3.2"),
)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

chain.invoke({"topic": "bears"})

chain.with_config(configurable={"llm": "ollama"}).invoke({"topic": "bears"})

chain.with_config(configurable={"llm": "anthropic"}).invoke({"topic": "bears"})

"""
### With Prompts

We can do a similar thing, but alternate between prompts
"""
logger.info("### With Prompts")

llm = ChatOllama(model="llama3.2")
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)
chain = prompt | llm

chain.invoke({"topic": "bears"})

chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})

"""
### With Prompts and LLMs

We can also have multiple things configurable!
Here's an example doing that with both prompts and LLMs.
"""
logger.info("### With Prompts and LLMs")

llm = ChatOllama(
    model="claude-3-haiku-20240307", temperature=0
).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="anthropic",
    ollama=ChatOllama(model="llama3.2"),
    gpt4=ChatOllama(model="llama3.2"),
)
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)
chain = prompt | llm

chain.with_config(configurable={"prompt": "poem", "llm": "ollama"}).invoke(
    {"topic": "bears"}
)

chain.with_config(configurable={"llm": "ollama"}).invoke({"topic": "bears"})

"""
### Saving configurations

We can also easily save configured chains as their own objects
"""
logger.info("### Saving configurations")

ollama_joke = chain.with_config(configurable={"llm": "ollama"})

ollama_joke.invoke({"topic": "bears"})

"""
## Next steps

You now know how to configure a chain's internal steps at runtime.

To learn more, see the other how-to guides on runnables in this section, including:

- Using [.bind()](/docs/how_to/binding) as a simpler way to set a runnable's runtime parameters
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)