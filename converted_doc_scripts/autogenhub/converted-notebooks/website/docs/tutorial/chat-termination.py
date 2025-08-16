from autogen import ConversableAgent
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Terminating Conversations Between Agents

In this chapter, we will explore how to terminate a conversation between AutoGen agents.

_But why is this important?_ Its because in any complex, autonomous workflows it's crucial to know when to stop the workflow. For example, when the task is completed, or perhaps when the process has consumed enough resources and needs to either stop or adopt different strategies, such as user intervention. So AutoGen natively supports several mechanisms to terminate conversations.

How to Control Termination with AutoGen?
Currently there are two broad mechanism to control the termination of conversations between agents:

1. **Specify parameters in `initiate_chat`**: When initiating a chat, you can define parameters that determine when the conversation should end.

2. **Configure an agent to trigger termination**: When defining individual agents, you can specify parameters that allow agents to terminate of a conversation based on particular (configurable) conditions.

## Parameters in `initiate_chat`
In the previous chapter we actually demonstrated this when we used the `max_turns` parameter to limit the number of turns. If we increase `max_turns` to say `3` notice the conversation takes more rounds to terminate:
"""
logger.info("# Terminating Conversations Between Agents")



cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
#     llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
)

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
#     llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)

result = joe.initiate_chat(
    cathy, message="Cathy, tell me a joke.", max_turns=3
)  # increase the number of max turns before termination

"""
## Agent-triggered termination
You can also terminate a conversation by configuring parameters of an agent.
Currently, there are two parameters you can configure:

1. `max_consecutive_auto_reply`: This condition triggers termination if the number of automatic responses to the same sender exceeds a threshold. You can customize this using the `max_consecutive_auto_reply` argument of the `ConversableAgent` class. To accomplish this the agent maintains a counter of the number of consecutive automatic responses to the same sender. Note that this counter can be reset because of human intervention. We will describe this in more detail in the next chapter.
2. `is_termination_msg`: This condition can trigger termination if the _received_ message satisfies a particular condition, e.g., it contains the word "TERMINATE". You can customize this condition using the `is_terminate_msg` argument in the constructor of the `ConversableAgent` class.

### Using `max_consecutive_auto_reply`

In the example below lets set `max_consecutive_auto_reply` to `1` and notice how this ensures that Joe only replies once.
"""
logger.info("## Agent-triggered termination")

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
#     llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    max_consecutive_auto_reply=1,  # Limit the number of consecutive auto-replies.
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.")

"""
### Using `is_termination_msg`

Let's set the termination message to "GOOD BYE" and see how the conversation terminates.
"""
logger.info("### Using `is_termination_msg`")

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
#     llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke and then say the words GOOD BYE.")

"""
Notice how the conversation ended based on contents of cathy's message!

## Summary

In this chapter we introduced mechanisms to terminate a conversation between agents.
You can configure both parameters in `initiate_chat` and also configuration of agents.

That said, it is important to note that when a termination condition is triggered,
the conversation may not always terminate immediately. The actual termination
depends on the `human_input_mode` argument of the `ConversableAgent` class.
For example, when mode is `NEVER` the termination conditions above will end the conversations.
But when mode is `ALWAYS` or `TERMINATE`, it will not terminate immediately.
We will describe this behavior and explain why it is important in the next chapter.
"""
logger.info("## Summary")

logger.info("\n\n[DONE]", bright=True)