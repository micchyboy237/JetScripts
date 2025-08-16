from jet.logger import CustomLogger
import autogen
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Task Solving with Code Generation, Execution, Debugging & Human Feedback

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to solve a challenging math problem with human feedback. Here `AssistantAgent` is an LLM-based agent that can write Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for a user to execute the code written by `AssistantAgent`. By setting `human_input_mode` properly, the `UserProxyAgent` can also prompt the user for feedback to `AssistantAgent`. For example, when `human_input_mode` is set to "ALWAYS", the `UserProxyAgent` will always prompt the user for feedback. When user feedback is provided, the `UserProxyAgent` will directly pass the feedback to `AssistantAgent`. When no user feedback is provided, the `UserProxyAgent` will execute the code written by `AssistantAgent` and return the execution results (success or failure and corresponding outputs) to `AssistantAgent`.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
```bash
pip install pyautogen
```
"""
logger.info("# Auto Generated Agent Chat: Task Solving with Code Generation, Execution, Debugging & Human Feedback")



"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("## Set your API Endpoint")



config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

"""
It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). Only the models with matching names are kept in the list based on the filter condition.

The config list looks like the following:
```python
config_list = [
    {
        'model': 'gpt-4',
        'api_key': '<your Ollama API key here>',
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

## Construct Agents

We construct the assistant agent and the user proxy agent.
"""
logger.info("## Construct Agents")

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 41,
        "config_list": config_list,
    },
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

"""
## Perform a task

We invoke the `initiate_chat()` method of the user proxy agent to start the conversation. When you run the cell below, you will be prompted to provide feedback after receiving a message from the assistant agent. If you don't provide any feedback (by pressing Enter directly), the user proxy agent will try to execute the code suggested by the assistant agent on behalf of you, or terminate if the assistant agent sends a "TERMINATE" signal at the end of the message.
"""
logger.info("## Perform a task")

math_problem_to_solve = """
Find $a + b + c$, given that $x+y \\neq -1$ and
\\begin{align}
	ax + by + c & = x + 7,\\
	a + bx + cy & = 2x + 6y,\\
	ay + b + cx & = 4x + y.
\\end{align}.
"""

user_proxy.initiate_chat(assistant, message=math_problem_to_solve)

"""
## Analyze the conversation

The human user can provide feedback at each step. When the human user didn't provide feedback, the code was executed. The executed results and error messages are returned to the assistant, and the assistant is able to modify the code based on the feedback. In the end, the task is complete and a "TERMINATE" signal is sent from the assistant. The user skipped feedback in the end and the conversation is finished.

After the conversation is finished, we can save the conversations between the two agents. The conversation can be accessed from `user_proxy.chat_messages`.
"""
logger.info("## Analyze the conversation")

logger.debug(user_proxy.chat_messages[assistant])

json.dump(user_proxy.chat_messages[assistant], open("conversations.json", "w"), indent=2)

logger.info("\n\n[DONE]", bright=True)