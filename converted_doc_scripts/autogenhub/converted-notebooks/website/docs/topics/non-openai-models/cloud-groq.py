from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated
import autogen
import json
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Groq

[Groq](https://groq.com/) is a cloud based platform serving a number of popular open weight models at high inference speeds. Models include Meta's Llama 3, Mistral AI's Mixtral, and Google's Gemma.

Although Groq's API is aligned well with Ollama's, which is the native API used by AutoGen, this library provides the ability to set specific parameters as well as track API costs.

You will need a Groq account and create an API key. [See their website for further details](https://groq.com/).

Groq provides a number of models to use, included below. See the list of [models here (requires login)](https://console.groq.com/docs/models).

See the sample `OAI_CONFIG_LIST` below showing how the Groq client class is used by specifying the `api_type` as `groq`.

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
        "model": "llama3-8b-8192",
        "api_key": "your Groq API Key goes here",
        "api_type": "groq"
    },
    {
        "model": "llama3-70b-8192",
        "api_key": "your Groq API Key goes here",
        "api_type": "groq"
    },
    {
        "model": "Mixtral 8x7b",
        "api_key": "your Groq API Key goes here",
        "api_type": "groq"
    },
    {
        "model": "gemma-7b-it",
        "api_key": "your Groq API Key goes here",
        "api_type": "groq"
    }
]
```

As an alternative to the `api_key` key and value in the config, you can set the environment variable `GROQ_API_KEY` to your Groq key.

Linux/Mac:
``` bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Windows:
``` bash
set GROQ_API_KEY=your_groq_api_key_here
```

## API parameters

The following parameters can be added to your config for the Groq API. See [this link](https://console.groq.com/docs/text-chat) for further information on them.

- frequency_penalty (number 0..1)
- max_tokens (integer >= 0)
- presence_penalty (number -2..2)
- seed (integer)
- temperature (number 0..2)
- top_p (number)

Example:
```python
[
    {
        "model": "llama3-8b-8192",
        "api_key": "your Groq API Key goes here",
        "api_type": "groq",
        "frequency_penalty": 0.5,
        "max_tokens": 2048,
        "presence_penalty": 0.2,
        "seed": 42,
        "temperature": 0.5,
        "top_p": 0.2
    }
]
```

## Two-Agent Coding Example

In this example, we run a two-agent chat with an AssistantAgent (primarily a coding agent) to generate code to count the number of prime numbers between 1 and 10,000 and then it will be executed.

We'll use Meta's Llama 3 model which is suitable for coding.
"""
logger.info("# Groq")


config_list = [
    {
        "model": "llama3-8b-8192",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "api_type": "groq",
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
    name="Groq Assistant",
    system_message=system_message,
    llm_config={"config_list": config_list},
)

chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="Provide code to count the number of prime numbers from 1 to 10000.",
)

"""
## Tool Call Example

In this example, instead of writing code, we will show how we can use Meta's Llama 3 model to perform parallel tool calling, where it recommends calling more than one tool at a time, using Groq's cloud inference.

We'll use a simple travel agent assistant program where we have a couple of tools for weather and currency conversion.

We start by importing libraries and setting up our configuration to use Meta's Llama 3 model and the `groq` client class.
"""
logger.info("## Tool Call Example")




config_list = [
    {"api_type": "groq", "model": "llama3-8b-8192", "api_key": os.getenv("GROQ_API_KEY"), "cache_seed": None}
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
We pass through our customers message and run the chat.

Finally, we ask the LLM to summarise the chat and print that out.
"""
logger.info("We pass through our customers message and run the chat.")

res = user_proxy.initiate_chat(
    chatbot,
    message="What's the weather in New York and can you tell me how much is 123.45 EUR in USD so I can spend it on my holiday? Throw a few holiday tips in as well.",
    summary_method="reflection_with_llm",
)

logger.debug(f"LLM SUMMARY: {res.summary['content']}")

"""
Using its fast inference, Groq required less than 2 seconds for the whole chat!

Additionally, Llama 3 was able to call both tools and pass through the right parameters. The `user_proxy` then executed them and this was passed back for Llama 3 to summarise the whole conversation.
"""
logger.info("Using its fast inference, Groq required less than 2 seconds for the whole chat!")

logger.info("\n\n[DONE]", bright=True)