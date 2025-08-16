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
# Together.AI

[Together.AI](https://www.together.ai/) is a cloud based platform serving many open-weight LLMs such as Google's Gemma, Meta's Llama 2/3, Qwen, Mistral.AI's Mistral/Mixtral, and NousResearch's Hermes models.

Although AutoGen can be used with Together.AI's API directly by changing the `base_url` to their url, it does not cater for some differences between messaging and it is recommended to use the Together.AI Client class as shown in this notebook.

You will need a Together.AI account and create an API key. [See their website for further details](https://www.together.ai/).

## Features

When using this client class, messages are tailored to accommodate the specific requirements of Together.AI's API and provide native support for function/tool calling, token usage, and accurate costs (as of June 2024).

## Getting started

First, you need to install the `pyautogen` package to use AutoGen with the Together.AI API library.

``` bash
pip install pyautogen[together]
```

Together.AI provides a large number of models to use, included some below. See the list of [models here](https://docs.together.ai/docs/inference-models).

See the sample `OAI_CONFIG_LIST` below showing how the Together.AI client class is used by specifying the `api_type` as `together`.

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
        "model": "google/gemma-7b-it",
        "api_key": "your Together.AI API Key goes here",
        "api_type": "together"
    },
    {
        "model": "codellama/CodeLlama-70b-Instruct-hf",
        "api_key": "your Together.AI API Key goes here",
        "api_type": "together"
    },
    {
        "model": "meta-llama/Llama-2-13b-chat-hf",
        "api_key": "your Together.AI API Key goes here",
        "api_type": "together"
    },
    {
        "model": "Qwen/Qwen2-72B-Instruct",
        "api_key": "your Together.AI API Key goes here",
        "api_type": "together"
    }
]
```

As an alternative to the `api_key` key and value in the config, you can set the environment variable `TOGETHER_API_KEY` to your Together.AI key.

Linux/Mac:
``` bash
export TOGETHER_API_KEY="your_together_ai_api_key_here"
```

Windows:
``` bash
set TOGETHER_API_KEY=your_together_ai_api_key_here
```

## API parameters

The following Together.AI parameters can be added to your config. See [this link](https://docs.together.ai/reference/chat-completions) for further information on their purpose, default values, and ranges.

- max_tokens (integer)
- temperature (float)
- top_p (float)
- top_k (integer)
- repetition_penalty (float)
- frequency_penalty (float)
- presence_penalty (float)
- min_p (float)
- safety_model (string - [moderation models here](https://docs.together.ai/docs/inference-models#moderation-models))

Example:
```python
[
    {
        "model": "microsoft/phi-2",
        "api_key": "your Together.AI API Key goes here",
        "api_type": "together",
        "max_tokens": 1000,
        "stream": False,
        "temperature": 1,
        "top_p": 0.8,
        "top_k": 50,
        "repetition_penalty": 0.5,
        "presence_penalty": 1.5,
        "frequency_penalty": 1.5,
        "min_p": 0.2,
        "safety_model": "Meta-Llama/Llama-Guard-7b"
    }
]
```

## Two-Agent Coding Example

In this example, we run a two-agent chat with an AssistantAgent (primarily a coding agent) to generate code to count the number of prime numbers between 1 and 10,000 and then it will be executed.

We'll use Mistral's Mixtral-8x7B instruct model which is suitable for coding.
"""
logger.info("# Together.AI")


config_list = [
    {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "api_type": "together",
        "stream": False,
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
    name="Together Assistant",
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

**Important:**

We are utilising a parameter that's supported by certain client classes, such as this one, called `hide_tools`. This parameter will hide the tools from the Together.AI response creation call if tools have already been executed and this helps minimise the chance of the LLM choosing a tool when we don't need it to.

Here we are using `if_all_run`, indicating that we want to hide the tools if all the tools have already been run. This will apply in each nested chat, so each time a player takes a turn it will aim to run both functions and then finish with a text response so we can hand control back to the other player.
"""
logger.info("## Tool Call Example")

config_list = [
    {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "api_type": "together",
        "hide_tools": "if_all_run",
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