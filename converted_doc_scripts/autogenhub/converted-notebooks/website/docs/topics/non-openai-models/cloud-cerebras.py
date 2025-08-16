from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.oai.cerebras import CerebrasClient, calculate_cerebras_cost
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated
import autogen
import json
import os
import time

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Cerebras

[Cerebras](https://cerebras.ai) has developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). Notably, the CS-3 system can run large language models like Llama-3.1-8B and Llama-3.1-70B at extremely fast speeds, making it an ideal platform for demanding AI workloads.

While it's technically possible to adapt AutoGen to work with Cerebras' API by updating the `base_url`, this approach may not fully account for minor differences in parameter support. Using this library will also allow for tracking of the API costs based on actual token usage.

For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai). Their API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai).

## Requirements
To use Cerebras with AutoGen, install the `autogen[cerebras]` package.
"""
logger.info("# Cerebras")

# !pip install autogen["cerebras"]

"""
## Getting Started

Cerebras provides a number of models to use. See the list of [models here](https://inference-docs.cerebras.ai/introduction).

See the sample `OAI_CONFIG_LIST` below showing how the Cerebras client class is used by specifying the `api_type` as `cerebras`.
```python
[
    {
        "model": "llama3.1-8b",
        "api_key": "your Cerebras API Key goes here",
        "api_type": "cerebras"
    },
    {
        "model": "llama3.1-70b",
        "api_key": "your Cerebras API Key goes here",
        "api_type": "cerebras"
    }
]
```

## Credentials

Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:

```
export CEREBRAS_API_KEY="your-api-key-here"
```

## API parameters

The following parameters can be added to your config for the Cerebras API. See [this link](https://inference-docs.cerebras.ai/api-reference/chat-completions) for further information on them and their default values.

- max_tokens (null, integer >= 0)
- seed (number)
- stream (True or False)
- temperature (number 0..1.5)
- top_p (number)

Example:
```python
[
    {
        "model": "llama3.1-70b",
        "api_key": "your Cerebras API Key goes here",
        "api_type": "cerebras"
        "max_tokens": 10000,
        "seed": 1234,
        "stream" True,
        "temperature": 0.5,
        "top_p": 0.2, # Note: It is recommended to set temperature or top_p but not both.
    }
]
```

## Two-Agent Coding Example

In this example, we run a two-agent chat with an AssistantAgent (primarily a coding agent) to generate code to count the number of prime numbers between 1 and 10,000 and then it will be executed.

We'll use Meta's LLama-3.1-70B model which is suitable for coding.
"""
logger.info("## Getting Started")



config_list = [{"model": "llama3.1-70b", "api_key": os.environ.get("CEREBRAS_API_KEY"), "api_type": "cerebras"}]

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
    name="Cerebras Assistant",
    system_message=system_message,
    llm_config={"config_list": config_list},
)

chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="Provide code to count the number of prime numbers from 1 to 10000.",
)

"""
## Tool Call Example

In this example, instead of writing code, we will show how Meta's Llama-3.1-70B model can perform parallel tool calling, where it recommends calling more than one tool at a time.

We'll use a simple travel agent assistant program where we have a couple of tools for weather and currency conversion.

We start by importing libraries and setting up our configuration to use Llama-3.1-70B and the `cerebras` client class.
"""
logger.info("## Tool Call Example")




config_list = [
    {
        "model": "llama3.1-70b",
        "api_key": os.environ.get("CEREBRAS_API_KEY"),
        "api_type": "cerebras",
    }
]

"""
Create our two agents.
"""
logger.info("Create our two agents.")

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""
        For currency exchange and weather forecasting tasks,
        only use the functions you have been provided with.
        When you summarize, make sure you've considered ALL previous instructions.
        Output 'HAVE FUN!' when an answer has been provided.
    """,
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


start_time = time.time()

res = user_proxy.initiate_chat(
    chatbot,
    message="What's the weather in New York and can you tell me how much is 123.45 EUR in USD so I can spend it on my holiday? Throw a few holiday tips in as well.",
    summary_method="reflection_with_llm",
)

end_time = time.time()

logger.debug(f"LLM SUMMARY: {res.summary['content']}\n\nDuration: {(end_time - start_time) * 1000}ms")

"""
We can see that the Cerebras Wafer-Scale Engine-3 (WSE-3) completed the query in 74ms -- faster than the blink of an eye!
"""
logger.info("We can see that the Cerebras Wafer-Scale Engine-3 (WSE-3) completed the query in 74ms -- faster than the blink of an eye!")

logger.info("\n\n[DONE]", bright=True)