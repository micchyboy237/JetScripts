from autogen.agentchat import AssistantAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from jet.logger import CustomLogger
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Group Chat with GPTAssistantAgent

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to get multiple `GPTAssistantAgent` converse through group chat.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
````{=mdx}
:::info Requirements
Install `autogen`:
```bash
pip install autogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Auto Generated Agent Chat: Group Chat with GPTAssistantAgent")


config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview", "gpt-4-32k"],
    },
)

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Define GPTAssistantAgent and GroupChat
"""
logger.info("## Define GPTAssistantAgent and GroupChat")

llm_config = {"config_list": config_list_gpt4, "cache_seed": 45}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="TERMINATE",
)

coder = GPTAssistantAgent(
    name="Coder",
    llm_config={
        "config_list": config_list_gpt4,
    },
    instructions=AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
)

analyst = GPTAssistantAgent(
    name="Data_analyst",
    instructions="You are a data analyst that offers insight into data.",
    llm_config={
        "config_list": config_list_gpt4,
    },
)
groupchat = autogen.GroupChat(agents=[user_proxy, coder, analyst], messages=[], max_round=10)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

"""
## Initiate Group Chat
Now all is set, we can initiate group chat.
"""
logger.info("## Initiate Group Chat")

user_proxy.initiate_chat(
    manager,
    message="Get the number of issues and pull requests for the repository 'autogenhub/autogen' over the past three weeks and offer analysis to the data. You should print the data in csv format grouped by weeks.",
)

logger.info("\n\n[DONE]", bright=True)