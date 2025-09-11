from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.globals import set_debug
from langchain.globals import set_verbose
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
import ChatModelTabs from "@theme/ChatModelTabs";
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
# How to debug your LLM apps

Like building any type of software, at some point you'll need to debug when building with LLMs. A model call will fail, or model output will be misformatted, or there will be some nested model calls and it won't be clear where along the way an incorrect output was created.

There are three main methods for debugging:

- Verbose Mode: This adds print statements for "important" events in your chain.
- Debug Mode: This add logging statements for ALL events in your chain.
- LangSmith Tracing: This logs events to [LangSmith](https://docs.smith.langchain.com/) to allow for visualization there.

|                        | Verbose Mode | Debug Mode | LangSmith Tracing |
|------------------------|--------------|------------|-------------------|
| Free                   | ✅            | ✅          | ✅                 |
| UI                     | ❌            | ❌          | ✅                 |
| Persisted              | ❌            | ❌          | ✅                 |
| See all events         | ❌            | ✅          | ✅                 |
| See "important" events | ✅            | ❌          | ✅                 |
| Runs Locally           | ✅            | ✅          | ❌                 |


## Tracing

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

Let's suppose we have an agent, and want to visualize the actions it takes and tool outputs it receives. Without any debugging, here's what we see:


<ChatModelTabs
  customVarName="llm"
/>
"""
logger.info("# How to debug your LLM apps")


llm = ChatOllama(model="llama3.2")


tools = [TavilySearch(max_results=5, topic="general")]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)

"""
We don't get much output, but since we set up LangSmith we can easily see what happened under the hood:

https://smith.langchain.com/public/a89ff88f-9ddc-4757-a395-3a1b365655bf/r

## `set_debug` and `set_verbose`

If you're prototyping in Jupyter Notebooks or running Python scripts, it can be helpful to print out the intermediate steps of a chain run.

There are a number of ways to enable printing at varying degrees of verbosity.

Note: These still work even with LangSmith enabled, so you can have both turned on and running at the same time

### `set_verbose(True)`

Setting the `verbose` flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.
"""
logger.info("## `set_debug` and `set_verbose`")


set_verbose(True)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)

"""
### `set_debug(True)`

Setting the global `debug` flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.
"""
logger.info("### `set_debug(True)`")


set_debug(True)
set_verbose(False)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke(
    {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
)

logger.info("\n\n[DONE]", bright=True)