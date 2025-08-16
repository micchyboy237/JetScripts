from jet.logger import CustomLogger
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
# Auto Generated Agent Chat: Collaborative Task Solving with Coding and Planning Agent

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to use multiple agents to work together and accomplish a task that requires finding info from the web and coding. `AssistantAgent` is an LLM-based agent that can write and debug Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for a user to execute the code written by `AssistantAgent`. We further create a planning agent for the assistant agent to consult. The planning agent is a variation of the LLM-based `AssistantAgent` with a different system message.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install autogen and docker:
```bash
pip install autogen docker
```
"""
logger.info("# Auto Generated Agent Chat: Collaborative Task Solving with Coding and Planning Agent")



"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file. It first looks for an environment variable with a specified name. The value of the environment variable needs to be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by filter_dict.

It's OK to have only the Ollama API key, or only the Azure Ollama API key + base.
"""
logger.info("## Set your API Endpoint")


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

"""
The config list looks like the following:
```python
config_list = [
    {
        'model': 'gpt-4',
        'api_key': '<your Ollama API key here>',
    },  # Ollama API endpoint for gpt-4
    {
        'model': 'gpt-4',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },  # Azure Ollama API endpoint for gpt-4
    {
        'model': 'gpt-4-32k',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },  # Azure Ollama API endpoint for gpt-4-32k
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/autogenhub/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods.

## Construct Agents

We construct the planning agent named "planner" and a user proxy agent for the planner named "planner_user". We specify `human_input_mode` as "NEVER" in the user proxy agent, which will never ask for human feedback. We define `ask_planner` function to send a message to the planner and return the suggestion from the planner.
"""
logger.info("## Construct Agents")

planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step that can be implemented by writing code. For example, browsing the web can be implemented by writing code that reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix.",
)
planner_user = autogen.UserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    return planner_user.last_message()["content"]

"""
We construct the assistant agent and the user proxy agent. We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for feedback when it receives a "TERMINATE" signal from the assistant agent. We set the `functions` in `AssistantAgent` and `function_map` in `UserProxyAgent` to use the created `ask_planner` function.
"""
logger.info("We construct the assistant agent and the user proxy agent. We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for feedback when it receives a "TERMINATE" signal from the assistant agent. We set the `functions` in `AssistantAgent` and `function_map` in `UserProxyAgent` to use the created `ask_planner` function.")

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    },
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "planning",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    function_map={"ask_planner": ask_planner},
)

"""
## Perform a task

We invoke the `initiate_chat()` method of the user proxy agent to start the conversation. When you run the cell below, you will be prompted to provide feedback after the assistant agent sends a "TERMINATE" signal at the end of the message. If you don't provide any feedback (by pressing Enter directly), the conversation will finish. Before the "TERMINATE" signal, the user proxy agent will try to execute the code suggested by the assistant agent on behalf of the user.
"""
logger.info("## Perform a task")

user_proxy.initiate_chat(
    assistant,
    message="""Suggest a fix to an open good first issue of flaml""",
)

"""
When the assistant needs to consult the planner, it suggests a function call to `ask_planner`. When this happens, a line like the following will be displayed:

***** Suggested function Call: ask_planner *****
"""
logger.info("When the assistant needs to consult the planner, it suggests a function call to `ask_planner`. When this happens, a line like the following will be displayed:")

logger.info("\n\n[DONE]", bright=True)