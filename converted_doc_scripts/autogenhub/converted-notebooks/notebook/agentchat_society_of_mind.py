from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent  # noqa: E402
from jet.logger import CustomLogger
import autogen  # noqa: E402
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# SocietyOfMindAgent

TODO: reimplement the society of mind agent using `register_nested_chat` and move to the core library.

This notebook demonstrates the SocietyOfMindAgent, which runs a group chat as an internal monologue, but appears to the external world as a single agent. This confers three distinct advantages:

1. It provides a clean way of producing a hierarchy of agents, hiding complexity as inner monologues.
2. It provides a consistent way of extracting an answer from a lengthy group chat (normally, it is not clear which message is the final response, and the response itself may not always be formatted in a way that makes sense when extracted as a standalone message).
3. It provides a way of recovering when agents exceed their context window constraints (the inner monologue is protected by try-catch blocks)

````{=mdx}
:::info Requirements
Install `autogen`:
```bash
pip install autogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# SocietyOfMindAgent")


llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={"model": ["codellama"]},
    ),
    "temperature": 0,
}
llm_config

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

### Example Group Chat with Two Agents

In this example, we will use an AssistantAgent and a UserProxy agent (configured for code execution) to work together to solve a problem. Executing code requires *at least* two conversation turns (one to write the code, and one to execute the code). If the code fails, or needs further refinement, then additional turns may also be needed. We will then wrap these agents in a SocietyOfMindAgent, hiding the internal discussion from other agents (though will still appear in the console), and ensuring that the response is suitable as a standalone message.

#### Construct the Inner-Monologue Agents
We begin by constructing the inner-monologue agents. These are the agents that do that real work.
"""
logger.info("### Example Group Chat with Two Agents")

assistant = autogen.AssistantAgent(
    "inner-assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

code_interpreter = autogen.UserProxyAgent(
    "inner-code-interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

groupchat = autogen.GroupChat(
    agents=[assistant, code_interpreter],
    messages=[],
    speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
    allow_repeat_speaker=False,
    max_round=8,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
)

"""
#### Construct and Run the SocietyOfMind Agent
We now wrap the inner group-chat with the SocietyOfMind Agent, and create a UserProxy to talk to it.
"""
logger.info("#### Construct and Run the SocietyOfMind Agent")


task = "On which days in 2024 was Microsoft Stock higher than $370?"

society_of_mind_agent = SocietyOfMindAgent(
    "society_of_mind",
    chat_manager=manager,
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: True,
)

user_proxy.initiate_chat(society_of_mind_agent, message=task)

"""
#### Remarks

There are a few things to notice about this output:
- First, the user_proxy sent only one message to the society_of_mind agent, and received only one message in response. As far as it is concerned, the society_of_mind agent is the only agent in the chat.
- Second, the final response is formatted in a way that is standalone. Unlike the prior response, it makes no reference of a previous script or execution, and it lacks the TERMINATE keyword that ended the inner monologue.
"""
logger.info("#### Remarks")

logger.info("\n\n[DONE]", bright=True)