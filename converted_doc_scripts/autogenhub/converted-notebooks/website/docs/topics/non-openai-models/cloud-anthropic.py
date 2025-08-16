from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from jet.logger import CustomLogger
from typing_extensions import Annotated
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Anthropic Claude

In the v0.2.30 release of AutoGen we support Anthropic Client.

Claude is a family of large language models developed by Anthropic and designed to revolutionize the way you interact with AI. Claude excels at a wide variety of tasks involving language, reasoning, analysis, coding, and more. The models are highly capable, easy to use, and can be customized to suit your needs.

In this notebook, we demonstrate how to use Anthropic Claude model for AgentChat in AutoGen.

## Features

Additionally, this client class provides support for function/tool calling and will track token usage and cost correctly as per Anthropic's API costs (as of June 2024).

## Requirements
To use Anthropic Claude with AutoGen, first you need to install the `pyautogen[anthropic]` package.

To try out the function call feature of Claude model, you need to install `anthropic>=0.23.1`.
"""
logger.info("# Anthropic Claude")

# !pip install pyautogen["anthropic"]

"""
## Set the config for the Anthropic API

You can add any parameters that are needed for the custom model loading in the same configuration list.

It is important to add the `api_type` field and set it to a string that corresponds to the client type used: `anthropic`.

Example:
```
[
    {
        "model": "claude-3-5-sonnet-20240620",
        "api_key": "your Anthropic API Key goes here",
        "api_type": "anthropic",
    },
    {
        "model": "claude-3-sonnet-20240229",
        "api_key": "your Anthropic API Key goes here",
        "api_type": "anthropic",
        "temperature": 0.5,
        "top_p": 0.2, # Note: It is recommended to set temperature or top_p but not both.
        "max_tokens": 10000,
    },
    {
        "model":"claude-3-opus-20240229",
        "api_key":"your api key",
        "api_type":"anthropic",
    },
    {
        "model":"claude-2.0",
        "api_key":"your api key",
        "api_type":"anthropic",
    },
    {
        "model":"claude-2.1",
        "api_key":"your api key",
        "api_type":"anthropic",
    },
    {
        "model":"claude-3.0-haiku",
        "api_key":"your api key",
        "api_type":"anthropic",
    },
]
```

### Alternative

# As an alternative to the api_key key and value in the config, you can set the environment variable `ANTHROPIC_API_KEY` to your Anthropic API key.

Linux/Mac:
```
# export ANTHROPIC_API_KEY="your Anthropic API key here"
```
Windows:
```
# set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
"""
logger.info("## Set the config for the Anthropic API")




config_list_claude = [
    {
        "model": "claude-3-5-sonnet-20240620",
#         "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
    }
]

"""
## Two-agent Coding Example

### Construct Agents

Construct a simple conversation between a User proxy and an ConversableAgent based on Claude-3 model.
"""
logger.info("## Two-agent Coding Example")

assistant = autogen.AssistantAgent(
    "assistant",
    llm_config={
        "config_list": config_list_claude,
    },
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=1,
)

"""
### Initiate Chat
"""
logger.info("### Initiate Chat")

user_proxy.initiate_chat(
    assistant, message="Write a python program to print the first 10 numbers of the Fibonacci sequence."
)

"""
## Tool Call Example with the Latest Anthropic API 
Anthropic just announced that tool use is now supported in the Anthropic API. To use this feature, please install `anthropic>=0.23.1`.

### Register the function
"""
logger.info("## Tool Call Example with the Latest Anthropic API")

@user_proxy.register_for_execution()  # Decorator factory for registering a function to be executed by an agent
@assistant.register_for_llm(
    name="get_weather", description="Get the current weather in a given location."
)  # Decorator factory for registering a function to be used by an agent
def preprocess(location: Annotated[str, "The city and state, e.g. Toronto, ON."]) -> str:
    return "Absolutely cloudy and rainy"

user_proxy.initiate_chat(
    assistant,
    message="What's the weather in Toronto?",
)

"""
## Group Chat Example with both Claude and GPT Agents

### A group chat with GPT-4 as the judge
"""
logger.info("## Group Chat Example with both Claude and GPT Agents")


config_list_gpt4 = [
    {
        "model": "gpt-4",
#         "api_key": os.getenv("OPENAI_API_KEY"),
        "api_type": "openai",
    }
]


config_list_gpt35 = [
    {
        "model": "gpt-3.5-turbo",
#         "api_key": os.getenv("OPENAI_API_KEY"),
        "api_type": "openai",
    }
]

alice = AssistantAgent(
    "Openai_agent",
    system_message="You are from MLX. You make arguments to support your company's position.",
    llm_config={
        "config_list": config_list_gpt4,
    },
)

bob = autogen.AssistantAgent(
    "Anthropic_agent",
    system_message="You are from Anthropic. You make arguments to support your company's position.",
    llm_config={
        "config_list": config_list_claude,
    },
)

charlie = AssistantAgent(
    "Research_Assistant",
    system_message="You are a helpful assistant to research the latest news and headlines.",
    llm_config={
        "config_list": config_list_gpt35,
    },
)

dan = AssistantAgent(
    "Judge",
    system_message="You are a judge. You will evaluate the arguments and make a decision on which one is more convincing.",
    llm_config={
        "config_list": config_list_gpt4,
    },
)

code_interpreter = UserProxyAgent(
    "code-interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)


@code_interpreter.register_for_execution()  # Decorator factory for registering a function to be executed by an agent
@charlie.register_for_llm(
    name="get_headlines", description="Get the headline of a particular day."
)  # Decorator factory for registering a function to be used by an agent
def get_headlines(headline_date: Annotated[str, "Date in MMDDYY format, e.g., 06192024"]) -> str:
    mock_news = {
        "06202024": "MLX competitor Anthropic announces its most powerful AI yet.",
        "06192024": "MLX founder Sutskever sets up new AI company devoted to safe superintelligence.",
    }
    return mock_news.get(headline_date, "No news available for today.")


groupchat = GroupChat(
    agents=[alice, bob, charlie, dan, code_interpreter],
    messages=[],
    allow_repeat_speaker=False,
    max_round=10,
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "config_list": config_list_gpt4,
    },
)

task = "Analyze the potential of MLX and Anthropic to revolutionize the field of AI based on today's headlines. Today is 06202024."

user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
)

user_proxy.initiate_chat(manager, message=task)

"""
### Same group chat with Claude 3.5 Sonnet as the judge
"""
logger.info("### Same group chat with Claude 3.5 Sonnet as the judge")

dan = AssistantAgent(
    "Judge",
    system_message="You are a judge. You will evaluate the arguments and make a decision on which one is more convincing.",
    llm_config={
        "config_list": config_list_claude,
    },
)

groupchat = GroupChat(
    agents=[alice, bob, charlie, dan, code_interpreter],
    messages=[],
    allow_repeat_speaker=False,
    max_round=10,
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "config_list": config_list_gpt4,
    },
)

user_proxy.initiate_chat(manager, message=task)

logger.info("\n\n[DONE]", bright=True)