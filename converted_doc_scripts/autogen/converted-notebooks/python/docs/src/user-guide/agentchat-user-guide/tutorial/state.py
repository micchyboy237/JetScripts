from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from jet.logger import logger
import json
import os
import shutil


async def main():

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")

    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)

    """
    # Managing State 
    
    So far, we have discussed how to build components in a multi-agent application - agents, teams, termination conditions. In many cases, it is useful to save the state of these components to disk and load them back later. This is particularly useful in a web application where stateless endpoints respond to requests and need to load the state of the application from persistent storage.
    
    In this notebook, we will discuss how to save and load the state of agents, teams, and termination conditions. 
     
    
    ## Saving and Loading Agents
    
    We can get the state of an agent by calling {py:meth}`~autogen_agentchat.agents.AssistantAgent.save_state` method on 
    an {py:class}`~autogen_agentchat.agents.AssistantAgent`.
    """
    logger.info("# Managing State")

    model_client = OllamaChatCompletionClient(model="llama3.2")

    assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )

    response = await assistant_agent.on_messages(
        [TextMessage(content="Write a 3 line poem on lake tangayika",
                     source="user")], CancellationToken()
    )
    logger.success(format_json(response))
    logger.debug(response.chat_message)
    await model_client.close()

    agent_state = await assistant_agent.save_state()
    logger.success(format_json(agent_state))
    logger.debug(agent_state)

    model_client = OllamaChatCompletionClient(model="llama3.2")

    new_assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )
    await new_assistant_agent.load_state(agent_state)

    response = await new_assistant_agent.on_messages(
        [TextMessage(content="What was the last line of the previous poem you wrote",
                     source="user")], CancellationToken()
    )
    logger.success(format_json(response))
    logger.debug(response.chat_message)
    await model_client.close()

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

    model_client = OllamaChatCompletionClient(model="llama3.2")

    assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )
    agent_team = RoundRobinGroupChat(
        [assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))

    stream = agent_team.run_stream(
        task="Write a beautiful poem 3-line about lake tangayika")

    await Console(stream)

    team_state = await agent_team.save_state()
    logger.success(format_json(team_state))

    """
    If we reset the team (simulating instantiation of the team),  and ask the question `What was the last line of the poem you wrote?`, we see that the team is unable to accomplish this as there is no reference to the previous run.
    """
    logger.info("If we reset the team (simulating instantiation of the team),  and ask the question `What was the last line of the poem you wrote?`, we see that the team is unable to accomplish this as there is no reference to the previous run.")

    await agent_team.reset()
    stream = agent_team.run_stream(
        task="What was the last line of the poem you wrote?")
    await Console(stream)

    """
    Next, we load the state of the team and ask the same question. We see that the team is able to accurately return the last line of the poem it wrote.
    """
    logger.info("Next, we load the state of the team and ask the same question. We see that the team is able to accurately return the last line of the poem it wrote.")

    logger.debug(team_state)

    await agent_team.load_state(team_state)
    stream = agent_team.run_stream(
        task="What was the last line of the poem you wrote?")
    await Console(stream)

    """
    ## Persisting State (File or Database)
    
    In many cases, we may want to persist the state of the team to disk (or a database) and load it back later. State is a dictionary that can be serialized to a file or written to a database.
    """
    logger.info("## Persisting State (File or Database)")

    team_state_dir = f"{OUTPUT_DIR}/coding"
    os.makedirs(team_state_dir, exist_ok=True)

    with open(f"{team_state_dir}/team_state.json", "w") as f:
        json.dump(team_state, f)

    with open(f"{team_state_dir}/team_state.json", "r") as f:
        team_state = json.load(f)

    new_agent_team = RoundRobinGroupChat(
        [assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))
    await new_agent_team.load_state(team_state)
    stream = new_agent_team.run_stream(
        task="What was the last line of the poem you wrote?")
    await Console(stream)
    await model_client.close()

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
