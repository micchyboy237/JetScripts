

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Building a Multi-Agent Application with AutoGen and Chainlit

In this sample, we will demonstrate how to build simple chat interface that
interacts with an [AgentChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)
agent or a team, using [Chainlit](https://github.com/Chainlit/chainlit),
and support streaming messages.

## Installation

To run this sample, you will need to install the following packages:
"""
logger.info("# Building a Multi-Agent Application with AutoGen and Chainlit")

pip install -U chainlit autogen-agentchat "autogen-ext[openai]" pyyaml

"""
To use other model providers, you will need to install a different extra
for the `autogen-ext` package.
See the [Models documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html) for more information.


## Model Configuration

Create a configuration file named `model_config.yaml` to configure the model
you want to use. Use `model_config_template.yaml` as a template.

## Running the Agent Sample

The first sample demonstrate how to interact with a single AssistantAgent
from the chat interface.
"""
logger.info("## Model Configuration")

chainlit run app_agent.py -h

"""
You can use one of the starters. For example, ask "What the weather in Seattle?".

The agent will respond by first using the tools provided and then reflecting
on the result of the tool execution.

## Running the Team Sample

The second sample demonstrate how to interact with a team of agents from the
chat interface.
"""
logger.info("## Running the Team Sample")

chainlit run app_team.py -h

"""
You can use one of the starters. For example, ask "Write a poem about winter.".

The team is a RoundRobinGroupChat, so each agent will respond in turn.
There are two agents in the team: one is instructed to be generally helpful
and the other one is instructed to be a critic and provide feedback. 
The two agents will respond in round-robin fashion until
the 'APPROVE' is mentioned by the critic agent.

## Running the Team Sample with UserProxyAgent

The third sample demonstrate how to interact with a team of agents including
a [UserProxyAgent](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.agents.html#autogen_agentchat.agents.UserProxyAgent)
for approval or rejection.
"""
logger.info("## Running the Team Sample with UserProxyAgent")

chainlit run app_team_user_proxy.py -h

"""
You can use one of the starters. For example, ask "Write code to reverse a string.".

By default, the `UserProxyAgent` will request an input action from the user
to approve or reject the response from the team.
When the user approves the response, the `UserProxyAgent` will send a message
to the team containing the text "APPROVE", and the team will stop responding.


## Next Steps

There are a few ways you can extend this example:

- Try other [agents](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html).
- Try other [team](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html) types beyond the `RoundRobinGroupChat`.
- Explore custom agents that sent multimodal messages.
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)