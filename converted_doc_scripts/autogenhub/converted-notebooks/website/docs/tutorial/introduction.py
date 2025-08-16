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
# Introduction to AutoGen

Welcome! AutoGen is an open-source framework that leverages multiple _agents_ to enable complex workflows. This tutorial introduces basic concepts and building blocks of AutoGen.

## Why AutoGen?

> _The whole is greater than the sum of its parts._<br/>
> -**Aristotle**

While there are many definitions of agents, in AutoGen, an agent is an entity that can send messages, receive messages and generate a reply using models, tools, human inputs or a mixture of them.
This abstraction not only allows agents to model real-world and abstract entities, such as people and algorithms, but it also simplifies implementation of complex workflows as collaboration among agents.

Further, AutoGen is extensible and composable: you can extend a simple agent with customizable components and create workflows that can combine these agents and power a more sophisticated agent, resulting in implementations that are modular and easy to maintain.

Most importantly, AutoGen is developed by a vibrant community of researchers
and engineers. It incorporates the latest research in multi-agent systems
and has been used in many real-world applications, including agent platform,
advertising, AI employees, blog/article writing, blockchain, calculate burned areas by wildfires,
customer support, cybersecurity, data analytics, debate, education, finance, gaming, legal consultation,
research, robotics, sales/marketing, social simulation, software engineering,
software security, supply chain, t-shirt design, training data generation, Youtube service...

## Installation

The simplest way to install AutoGen is from pip: `pip install autogen`. Find more options in [Installation](/docs/installation/).

## Agents

In AutoGen, an agent is an entity that can send and receive messages to and from
other agents in its environment. An agent can be powered by models (such as a large language model
like GPT-4), code executors (such as an IPython kernel), human, or a combination of these
and other pluggable and customizable components.

```{=mdx}
![ConversableAgent](./assets/conversable-agent.jpg)
```

An example of such agents is the built-in `ConversableAgent` which supports the following components:

1. A list of LLMs
2. A code executor
3. A function and tool executor
4. A component for keeping human-in-the-loop

You can switch each component on or off and customize it to suit the need of 
your application. For advanced users, you can add additional components to the agent
by using [`registered_reply`](../reference/agentchat/conversable_agent/#register_reply).

LLMs, for example, enable agents to converse in natural languages and transform between structured and unstructured text. 
The following example shows a `ConversableAgent` with a GPT-4 LLM switched on and other
components switched off:
"""
logger.info("# Introduction to AutoGen")



agent = ConversableAgent(
    "chatbot",
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

"""
The `llm_config` argument contains a list of configurations for the LLMs.
See [LLM Configuration](/docs/topics/llm_configuration) for more details.

You can ask this agent to generate a response to a question using the `generate_reply` method:
"""
logger.info("The `llm_config` argument contains a list of configurations for the LLMs.")

reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
logger.debug(reply)

"""
## Roles and Conversations

In AutoGen, you can assign roles to agents and have them participate in conversations or chat with each other. A conversation is a sequence of messages exchanged between agents. You can then use these conversations to make progress on a task. For example, in the example below, we assign different roles to two agents by setting their
`system_message`.
"""
logger.info("## Roles and Conversations")

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

"""
Now that we have two comedian agents, we can ask them to start a comedy show.
This can be done using the `initiate_chat` method.
We set the `max_turns` to 2 to keep the conversation short.
"""
logger.info("Now that we have two comedian agents, we can ask them to start a comedy show.")

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)

"""
The comedians are bouncing off each other!

## Summary

In this chapter, we introduced the concept of agents, roles and conversations in AutoGen.
For simplicity, we only used LLMs and created fully autonomous agents (`human_input_mode` was set to `NEVER`). 
In the next chapter, 
we will show how you can control when to _terminate_ a conversation between autonomous agents.
"""
logger.info("## Summary")

logger.info("\n\n[DONE]", bright=True)