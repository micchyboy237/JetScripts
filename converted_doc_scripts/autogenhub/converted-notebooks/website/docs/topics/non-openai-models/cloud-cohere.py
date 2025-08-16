from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from jet.logger import CustomLogger
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated
import autogen
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Cohere

[Cohere](https://cohere.com/) is a cloud based platform serving their own LLMs, in particular the Command family of models.

Cohere's API differs from Ollama's, which is the native API used by AutoGen, so to use Cohere's LLMs you need to use this library.

You will need a Cohere account and create an API key. [See their website for further details](https://cohere.com/).

## Features

When using this client class, AutoGen's messages are automatically tailored to accommodate the specific requirements of Cohere's API.

Additionally, this client class provides support for function/tool calling and will track token usage and cost correctly as per Cohere's API costs (as of July 2024).

## Getting started

First you need to install the `pyautogen` package to use AutoGen with the Cohere API library.

``` bash
pip install pyautogen[cohere]
```

Cohere provides a number of models to use, included below. See the list of [models here](https://docs.cohere.com/docs/models).

See the sample `OAI_CONFIG_LIST` below showing how the Cohere client class is used by specifying the `api_type` as `cohere`.

```python
[
    {
        "model": "gpt-35-turbo",
        "api_key": "your Ollama Key goes here",
    },
    {
        "model": "gpt-4-vision-preview",
        "api_key": "your Ollama Key goes here",
    },
    {
        "model": "dalle",
        "api_key": "your Ollama Key goes here",
    },
    {
        "model": "command-r-plus",
        "api_key": "your Cohere API Key goes here",
        "api_type": "cohere"
    },
    {
        "model": "command-r",
        "api_key": "your Cohere API Key goes here",
        "api_type": "cohere"
    },
    {
        "model": "command",
        "api_key": "your Cohere API Key goes here",
        "api_type": "cohere"
    }
]
```

As an alternative to the `api_key` key and value in the config, you can set the environment variable `COHERE_API_KEY` to your Cohere key.

Linux/Mac:
``` bash
export COHERE_API_KEY="your_cohere_api_key_here"
```

Windows:
``` bash
set COHERE_API_KEY=your_cohere_api_key_here
```

## API parameters

The following parameters can be added to your config for the Cohere API. See [this link](https://docs.cohere.com/reference/chat) for further information on them and their default values.

- temperature (number > 0)
- p (number 0.01..0.99)
- k (number 0..500)
- max_tokens (null, integer >= 0)
- seed (null, integer)
- frequency_penalty (number 0..1)
- presence_penalty (number 0..1)
- client_name (null, string)

Example:
```python
[
    {
        "model": "command-r",
        "api_key": "your Cohere API Key goes here",
        "api_type": "cohere",
        "client_name": "autogen-cohere",
        "temperature": 0.5,
        "p": 0.2,
        "k": 100,
        "max_tokens": 2048,
        "seed": 42,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2
    }
]
```

## Two-Agent Coding Example

In this example, we run a two-agent chat with an AssistantAgent (primarily a coding agent) to generate code to count the number of prime numbers between 1 and 10,000 and then it will be executed.

We'll use Cohere's Command R model which is suitable for coding.
"""
logger.info("# Cohere")


config_list = [
    {
        "model": "command-r",
        "api_key": os.environ.get("COHERE_API_KEY"),
        "api_type": "cohere",
    }
]

"""
Importantly, we have tweaked the system message so that the model doesn't return the termination keyword, which we've changed to FINISH, with the code block.
"""
logger.info("Importantly, we have tweaked the system message so that the model doesn't return the termination keyword, which we've changed to FINISH, with the code block.")



workdir = Path("coding")
workdir.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=workdir)


user_proxy_agent = UserProxyAgent(
    name="User",
    code_execution_config={"executor": code_executor},
    is_termination_msg=lambda msg: "FINISH" in msg.get("content"),
)

system_message = """You are a helpful AI assistant who writes code and the user executes it.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) for the user to execute.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
IMPORTANT: Wait for the user to execute your code and then you can reply with the word "FINISH". DO NOT OUTPUT "FINISH" after your code block."""

assistant_agent = AssistantAgent(
    name="Cohere Assistant",
    system_message=system_message,
    llm_config={"config_list": config_list},
)

chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="Provide code to count the number of prime numbers from 1 to 10000.",
)

"""
## Tool Call Example

In this example, instead of writing code, we will show how Cohere's Command R+ model can perform parallel tool calling, where it recommends calling more than one tool at a time.

We'll use a simple travel agent assistant program where we have a couple of tools for weather and currency conversion.

We start by importing libraries and setting up our configuration to use Command R+ and the `cohere` client class.
"""
logger.info("## Tool Call Example")




config_list = [
    {"api_type": "cohere", "model": "command-r-plus", "api_key": os.getenv("COHERE_API_KEY"), "cache_seed": None}
]

"""
Create our two agents.
"""
logger.info("Create our two agents.")

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""For currency exchange and weather forecasting tasks,
        only use the functions you have been provided with.
        Output 'HAVE FUN!' when an answer has been provided.""",
    llm_config={"config_list": config_list},
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and "HAVE FUN!" in x.get("content", ""),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

"""
Create the two functions, annotating them so that those descriptions can be passed through to the LLM.

We associate them with the agents using `register_for_execution` for the user_proxy so it can execute the function and `register_for_llm` for the chatbot (powered by the LLM) so it can pass the function definitions to the LLM.
"""
logger.info("Create the two functions, annotating them so that those descriptions can be passed through to the LLM.")

CurrencySymbol = Literal["USD", "EUR"]



def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")




@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{format(quote_amount, '.2f')} {quote_currency}"




def get_current_weather(location, unit="fahrenheit"):
    """Get the weather for some location"""
    if "chicago" in location.lower():
        return json.dumps({"location": "Chicago", "temperature": "13", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "55", "unit": unit})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "11", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})




@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Weather forecast for US cities.")
def weather_forecast(
    location: Annotated[str, "City name"],
) -> str:
    weather_details = get_current_weather(location=location)
    weather = json.loads(weather_details)
    return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}"

"""
We pass through our customer's message and run the chat.

Finally, we ask the LLM to summarise the chat and print that out.
"""
logger.info("We pass through our customer's message and run the chat.")

res = user_proxy.initiate_chat(
    chatbot,
    message="What's the weather in New York and can you tell me how much is 123.45 EUR in USD so I can spend it on my holiday? Throw a few holiday tips in as well.",
    summary_method="reflection_with_llm",
)

logger.debug(f"LLM SUMMARY: {res.summary['content']}")

"""
We can see that Command R+ recommended we call both tools and passed through the right parameters. The `user_proxy` executed them and this was passed back to Command R+ to interpret them and respond. Finally, Command R+ was asked to summarise the whole conversation.
"""
logger.info("We can see that Command R+ recommended we call both tools and passed through the right parameters. The `user_proxy` executed them and this was passed back to Command R+ to interpret them and respond. Finally, Command R+ was asked to summarise the whole conversation.")

logger.info("\n\n[DONE]", bright=True)