import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Managing State 

So far, we have discussed how to build components in a multi-agent application - agents, teams, termination conditions. In many cases, it is useful to save the state of these components to disk and load them back later. This is particularly useful in a web application where stateless endpoints respond to requests and need to load the state of the application from persistent storage.

In this notebook, we will discuss how to save and load the state of agents, teams, and termination conditions. 
 

## Saving and Loading Agents

We can get the state of an agent by calling {py:meth}`~autogen_agentchat.agents.AssistantAgent.save_state` method on 
an {py:class}`~autogen_agentchat.agents.AssistantAgent`.
"""
logger.info("# Managing State")


model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
    model_client=model_client,
)

async def async_func_16():
    response = await assistant_agent.on_messages(
        [TextMessage(content="Write a 3 line poem on lake tangayika", source="user")], CancellationToken()
    )
    return response
response = asyncio.run(async_func_16())
logger.success(format_json(response))
logger.debug(response.chat_message)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

async def run_async_code_5d9ca148():
    async def run_async_code_87962345():
        agent_state = await assistant_agent.save_state()
        return agent_state
    agent_state = asyncio.run(run_async_code_87962345())
    logger.success(format_json(agent_state))
    return agent_state
agent_state = asyncio.run(run_async_code_5d9ca148())
logger.success(format_json(agent_state))
logger.debug(agent_state)

model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

new_assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
    model_client=model_client,
)
async def run_async_code_60b630b2():
    await new_assistant_agent.load_state(agent_state)
    return 
 = asyncio.run(run_async_code_60b630b2())
logger.success(format_json())

async def async_func_34():
    response = await new_assistant_agent.on_messages(
        [TextMessage(content="What was the last line of the previous poem you wrote", source="user")], CancellationToken()
    )
    return response
response = asyncio.run(async_func_34())
logger.success(format_json(response))
logger.debug(response.chat_message)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
```{note}
For {py:class}`~autogen_agentchat.agents.AssistantAgent`, its state consists of the model_context.
If you write your own custom agent, consider overriding the {py:meth}`~autogen_agentchat.agents.BaseChatAgent.save_state` and {py:meth}`~autogen_agentchat.agents.BaseChatAgent.load_state` methods to customize the behavior. The default implementations save and load an empty state.
```

## Saving and Loading Teams 

We can get the state of a team by calling `save_state` method on the team and load it back by calling `load_state` method on the team. 

When we call `save_state` on a team, it saves the state of all the agents in the team.

We will begin by creating a simple {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` team with a single agent and ask it to write a poem.
"""
logger.info("## Saving and Loading Teams")

model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
    model_client=model_client,
)
agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))

stream = agent_team.run_stream(task="Write a beautiful poem 3-line about lake tangayika")

async def run_async_code_71db6073():
    await Console(stream)
    return 
 = asyncio.run(run_async_code_71db6073())
logger.success(format_json())

async def run_async_code_3b76751f():
    async def run_async_code_caa5fd86():
        team_state = await agent_team.save_state()
        return team_state
    team_state = asyncio.run(run_async_code_caa5fd86())
    logger.success(format_json(team_state))
    return team_state
team_state = asyncio.run(run_async_code_3b76751f())
logger.success(format_json(team_state))

"""
If we reset the team (simulating instantiation of the team),  and ask the question `What was the last line of the poem you wrote?`, we see that the team is unable to accomplish this as there is no reference to the previous run.
"""
logger.info("If we reset the team (simulating instantiation of the team),  and ask the question `What was the last line of the poem you wrote?`, we see that the team is unable to accomplish this as there is no reference to the previous run.")

async def run_async_code_4e849c8a():
    await agent_team.reset()
    return 
 = asyncio.run(run_async_code_4e849c8a())
logger.success(format_json())
stream = agent_team.run_stream(task="What was the last line of the poem you wrote?")
async def run_async_code_71db6073():
    await Console(stream)
    return 
 = asyncio.run(run_async_code_71db6073())
logger.success(format_json())

"""
Next, we load the state of the team and ask the same question. We see that the team is able to accurately return the last line of the poem it wrote.
"""
logger.info("Next, we load the state of the team and ask the same question. We see that the team is able to accurately return the last line of the poem it wrote.")

logger.debug(team_state)

async def run_async_code_0a74f6a5():
    await agent_team.load_state(team_state)
    return 
 = asyncio.run(run_async_code_0a74f6a5())
logger.success(format_json())
stream = agent_team.run_stream(task="What was the last line of the poem you wrote?")
async def run_async_code_71db6073():
    await Console(stream)
    return 
 = asyncio.run(run_async_code_71db6073())
logger.success(format_json())

"""
## Persisting State (File or Database)

In many cases, we may want to persist the state of the team to disk (or a database) and load it back later. State is a dictionary that can be serialized to a file or written to a database.
"""
logger.info("## Persisting State (File or Database)")



with open("coding/team_state.json", "w") as f:
    json.dump(team_state, f)

with open("coding/team_state.json", "r") as f:
    team_state = json.load(f)

new_agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))
async def run_async_code_9386460e():
    await new_agent_team.load_state(team_state)
    return 
 = asyncio.run(run_async_code_9386460e())
logger.success(format_json())
stream = new_agent_team.run_stream(task="What was the last line of the poem you wrote?")
async def run_async_code_71db6073():
    await Console(stream)
    return 
 = asyncio.run(run_async_code_71db6073())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)