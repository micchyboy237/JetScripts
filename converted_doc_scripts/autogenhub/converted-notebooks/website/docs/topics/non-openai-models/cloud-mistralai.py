from IPython.display import display
from autogen import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent, register_function
from autogen.coding import LocalCommandLineCodeExecutor
from jet.logger import CustomLogger
from pathlib import Path
from typing_extensions import Annotated
import chess
import chess.svg
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Mistral AI

[Mistral AI](https://mistral.ai/) is a cloud based platform serving their own LLMs, like Mistral, Mixtral, and Codestral.

Although AutoGen can be used with Mistral AI's API directly by changing the `base_url` to their url, it does not cater for some differences between messaging and, with their API being more strict than MLX's, it is recommended to use the Mistral AI Client class as shown in this notebook.

You will need a Mistral.AI account and create an API key. [See their website for further details](https://mistral.ai/).

## Features

When using this client class, messages are automatically tailored to accommodate the specific requirements of Mistral AI's API (such as role orders), which have become more strict than MLX's API.

Additionally, this client class provides support for function/tool calling and will track token usage and cost correctly as per Mistral AI's API costs (as of June 2024).

## Getting started

First you need to install the `pyautogen` package to use AutoGen with the Mistral API library.

``` bash
pip install pyautogen[mistral]
```

Mistral provides a number of models to use, included below. See the list of [models here](https://docs.mistral.ai/platform/endpoints/).

See the sample `OAI_CONFIG_LIST` below showing how the Mistral AI client class is used by specifying the `api_type` as `mistral`.

```python
[
    {
        "model": "gpt-35-turbo",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "gpt-4-vision-preview",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "dalle",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "open-mistral-7b",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "open-mixtral-8x7b",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "open-mixtral-8x22b",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "mistral-small-latest",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "mistral-medium-latest",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "mistral-large-latest",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    },
    {
        "model": "codestral-latest",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral"
    }
]
```

As an alternative to the `api_key` key and value in the config, you can set the environment variable `MISTRAL_API_KEY` to your Mistral AI key.

Linux/Mac:
``` bash
export MISTRAL_API_KEY="your_mistral_ai_api_key_here"
```

Windows:
``` bash
set MISTRAL_API_KEY=your_mistral_ai_api_key_here
```

## API parameters

The following parameters can be added to your config for the Mistral.AI API. See [this link](https://docs.mistral.ai/api/#operation/createChatCompletion) for further information on them and their default values.

- temperature (number 0..1)
- top_p (number 0..1)
- max_tokens (null, integer >= 0)
- random_seed (null, integer)
- safe_prompt (True or False)

Example:
```python
[
    {
        "model": "codestral-latest",
        "api_key": "your Mistral AI API Key goes here",
        "api_type": "mistral",
        "temperature": 0.5,
        "top_p": 0.2, # Note: It is recommended to set temperature or top_p but not both.
        "max_tokens": 10000,
        "safe_prompt": False,
        "random_seed": 42
    }
]
```

## Two-Agent Coding Example

In this example, we run a two-agent chat with an AssistantAgent (primarily a coding agent) to generate code to count the number of prime numbers between 1 and 10,000 and then it will be executed.

We'll use Mistral's Mixtral 8x22B model which is suitable for coding.
"""
logger.info("# Mistral AI")


config_list = [
    {
        "model": "open-mixtral-8x22b",
        "api_key": os.environ.get("MISTRAL_API_KEY"),
        "api_type": "mistral",
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
    name="Mistral Assistant",
    system_message=system_message,
    llm_config={"config_list": config_list},
)

chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="Provide code to count the number of prime numbers from 1 to 10000.",
)

"""
## Tool Call Example

In this example, instead of writing code, we will have two agents playing chess against each other using tool calling to make moves.

We'll change models to Mistral AI's large model for this challenging task.
"""
logger.info("## Tool Call Example")

config_list = [
    {
        "model": "mistral-large-latest",
        "api_key": os.environ.get("MISTRAL_API_KEY"),
        "api_type": "mistral",
    }
]

"""
First install the `chess` package by running the following command:
"""
logger.info("First install the `chess` package by running the following command:")

# ! pip install chess

"""
Write the function for making a move.
"""
logger.info("Write the function for making a move.")



board = chess.Board()


def make_move() -> Annotated[str, "A move in UCI format"]:
    moves = list(board.legal_moves)
    move = random.choice(moves)
    board.push(move)
    display(chess.svg.board(board, size=400))
    return str(move)

"""
Let's create the agents. We have three different agents:
- `player_white` is the agent that plays white.
- `player_black` is the agent that plays black.
- `board_proxy` is the agent that moves the pieces on the board.
"""
logger.info("Let's create the agents. We have three different agents:")


player_white = ConversableAgent(
    name="Player White",
    system_message="You are a chess player and you play as white. " "Always call make_move() to make a move",
    llm_config={"config_list": config_list, "cache_seed": None},
)

player_black = ConversableAgent(
    name="Player Black",
    system_message="You are a chess player and you play as black. " "Always call make_move() to make a move",
    llm_config={"config_list": config_list, "cache_seed": None},
)

board_proxy = ConversableAgent(
    name="Board Proxy",
    llm_config=False,
    is_termination_msg=lambda msg: "tool_calls" not in msg,
)

"""
Register tools for the agents. See the [tutorial chapter on tool use](/docs/tutorial/tool-use) 
for more information.
"""
logger.info("Register tools for the agents. See the [tutorial chapter on tool use](/docs/tutorial/tool-use)")

register_function(
    make_move,
    caller=player_white,
    executor=board_proxy,
    name="make_move",
    description="Make a move.",
)

register_function(
    make_move,
    caller=player_black,
    executor=board_proxy,
    name="make_move",
    description="Make a move.",
)

"""
Register nested chats for the player agents.
Nested chats allows each player agent to chat with the board proxy agent
to make a move, before communicating with the other player agent.
See the [nested chats tutorial chapter](/docs/tutorial/conversation-patterns#nested-chats)
for more information.
"""
logger.info("Register nested chats for the player agents.")

player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
        }
    ],
)

"""
Clear the board and start the chess game.
"""
logger.info("Clear the board and start the chess game.")

board = chess.Board()

chat_result = player_white.initiate_chat(
    player_black,
    message="Let's play chess! Your move.",
    max_turns=4,
)

logger.info("\n\n[DONE]", bright=True)