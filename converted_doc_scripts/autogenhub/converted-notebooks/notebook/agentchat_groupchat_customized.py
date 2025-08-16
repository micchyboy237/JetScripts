from autogen import Agent
from jet.logger import CustomLogger
from typing import Dict, List
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
# Group Chat with Customized Speaker Selection Method

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to pass a cumstomized agent selection method to GroupChat. The customized function looks like this:

```python
def custom_speaker_selection_func(last_speaker, groupchat):
    """
logger.info("# Group Chat with Customized Speaker Selection Method")Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Parameters:
        - last_speaker: Agent
            The last speaker in the group chat.
        - groupchat: GroupChat
            The GroupChat object
    Return:
        Return one of the following:
        1. an `Agent` class, it must be one of the agents in the group chat.
        2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        3. None, which indicates the chat should be terminated.
    """
    pass

groupchat = autogen.GroupChat(
    speaker_selection_method=custom_speaker_selection_func,
    ...,
)
```
The last speaker and the groupchat object are passed to the function. Commonly used variables from groupchat are `groupchat.messages` an `groupchat.agents`, which is the message history and the agents in the group chat respectively. You can access other attributes of the groupchat, such as `groupchat.allowed_speaker_transitions_dict` for pre-defined allowed_speaker_transitions_dict. 

````{=mdx}
:::info Requirements
Install `pyautogen`:
```bash
pip install pyautogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("## Set your API Endpoint")


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview"],
    },
)

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Construct Agents

- Planner: Give a plan and revise.
- Admin: Human in the loop to approve or terminate the process.
- Engineer: Retrieve papers from the internet by writing code.
- Executor: Execute the code.
- Scientist: Read the papers and write a summary.

The pipeline is the following:

1. The planner interact with Admin (user) to revise a plan. Only when the Admin types "Approve", we can move to the next step.
2. The engineer will write code to retrieve papers from the internet. The code will be executed by executor.
3. When the code is executed successfully, the scientist will read the papers and write a summary.
4. The summary will be reviewed by the Admin and give comments. When the Admin types "TERMINATE", the process will be terminated.
"""
logger.info("## Construct Agents")

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=gpt4_config,
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)




def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        return planner

    if last_speaker is user_proxy:
        if "Approve" in messages[-1]["content"]:
            return engineer
        elif messages[-2]["name"] == "Planner":
            return planner
        elif messages[-2]["name"] == "Scientist":
            return scientist

    elif last_speaker is planner:
        return user_proxy

    elif last_speaker is engineer:
        if "```python" in messages[-1]["content"]:
            return executor
        else:
            return engineer

    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return engineer
        else:
            return scientist

    elif last_speaker is scientist:
        return user_proxy

    else:
        return "random"


groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

"""
## Start Chat
"""
logger.info("## Start Chat")

user_proxy.initiate_chat(
    manager, message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software."
)

logger.info("\n\n[DONE]", bright=True)