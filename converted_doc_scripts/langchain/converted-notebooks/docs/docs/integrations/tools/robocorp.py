from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, OllamaFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_robocorp import ActionServerToolkit
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
# Robocorp Toolkit

This notebook covers how to get started with [Robocorp Action Server](https://github.com/robocorp/robocorp) action toolkit and LangChain.

Robocorp is the easiest way to extend the capabilities of AI agents, assistants and copilots with custom actions.

## Installation

First, see the [Robocorp Quickstart](https://github.com/robocorp/robocorp#quickstart) on how to setup `Action Server` and create your Actions.

In your LangChain application, install the `langchain-robocorp` package:
"""
logger.info("# Robocorp Toolkit")

# %pip install --upgrade --quiet langchain-robocorp

"""
When you create the new `Action Server` following the above quickstart.

It will create a directory with files, including `action.py`.

We can add python function as actions as shown [here](https://github.com/robocorp/robocorp/tree/master/actions#describe-your-action).

Let's add a dummy function to `action.py`.

```python
@action
def get_weather_forecast(city: str, days: int, scale: str = "celsius") -> str:
    """
logger.info("When you create the new `Action Server` following the above quickstart.")
    Returns weather conditions forecast for a given city.

    Args:
        city (str): Target city to get the weather conditions for
        days: How many day forecast to return
        scale (str): Temperature scale to use, should be one of "celsius" or "fahrenheit"

    Returns:
        str: The requested weather conditions forecast
    """
    return "75F and sunny :)"
```

We then start the server:

```bash
action-server start
```

And we can see: 

```
Found new action: get_weather_forecast

```

Test locally by going to the server running at `http://localhost:8080` and use the UI to run the function.

## Environment Setup

Optionally you can set the following environment variables:

- `LANGSMITH_TRACING=true`: To enable LangSmith log run tracing that can also be bind to respective Action Server action run logs. See [LangSmith documentation](https://docs.smith.langchain.com/tracing#log-runs) for more.

## Usage

We started the local action server, above, running on `http://localhost:8080`.
"""
logger.info("## Environment Setup")


llm = ChatOllama(model="llama3.2")

toolkit = ActionServerToolkit(url="http://localhost:8080", report_trace=True)
tools = toolkit.get_tools()

system_message = SystemMessage(content="You are a helpful assistant")
prompt = OllamaFunctionsAgent.create_prompt(system_message)
agent = OllamaFunctionsAgent(llm=llm, prompt=prompt, tools=tools)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke("What is the current weather today in San Francisco in fahrenheit?")

"""
### Single input tools

By default `toolkit.get_tools()` will return the actions as Structured Tools. 

To return single input tools, pass a Chat model to be used for processing the inputs.
"""
logger.info("### Single input tools")

toolkit = ActionServerToolkit(url="http://localhost:8080")
tools = toolkit.get_tools(llm=llm)

logger.info("\n\n[DONE]", bright=True)