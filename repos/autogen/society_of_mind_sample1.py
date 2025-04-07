import asyncio
from typing import Optional

from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import AgentRuntime, SingleThreadedAgentRuntime
# from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient

from jet.transformers.object import make_serializable
from jet.transformers.formatters import format_json
from jet.logger import logger

model_client = OllamaChatCompletionClient(model="mistral")
model_client2 = OllamaChatCompletionClient(model="llama3.1")


async def society_of_mind_agent_multiple_rounds(runtime: Optional[AgentRuntime]):
    agent1 = AssistantAgent("assistant1", model_client=model_client,
                            system_message="You are a helpful assistant.")
    agent2 = AssistantAgent("assistant2", model_client=model_client2,
                            system_message="You are a helpful assistant.")
    team = RoundRobinGroupChat(
        [agent1, agent2], MaxMessageTermination(3), runtime=runtime)
    society = SocietyOfMindAgent(
        "society_of_mind", team=team, model_client=model_client)

    r1 = await society.run(task="Count to 10.")
    r2 = await society.run()
    r3 = await society.run()

    return {
        "round1": make_serializable(r1),
        "round2": make_serializable(r2),
        "round3": make_serializable(r3)
    }


async def society_of_mind_agent_basic(runtime: Optional[AgentRuntime]):
    agent1 = AssistantAgent("assistant1", model_client=model_client,
                            system_message="You are a helpful assistant.")
    agent2 = AssistantAgent("assistant2", model_client=model_client2,
                            system_message="You are a helpful assistant.")
    team = RoundRobinGroupChat(
        [agent1, agent2], MaxMessageTermination(3), runtime=runtime)

    society = SocietyOfMindAgent(
        "society_of_mind", team=team, model_client=model_client)
    response = await society.run(task="Count to 10.")

    state = await society.save_state()

    # Restore
    restored_team = RoundRobinGroupChat(
        [agent1, agent2], MaxMessageTermination(3), runtime=runtime)
    restored_society = SocietyOfMindAgent(
        "society_of_mind", team=restored_team, model_client=model_client)
    await restored_society.load_state(state)
    restored_state = await restored_society.save_state()

    config = society.dump_component()
    loaded = SocietyOfMindAgent.load_component(config)

    return {
        "response": make_serializable(response),
        "state_equal": state == restored_state,
        "config_provider": config.provider,
        "loaded_type_check": isinstance(loaded, SocietyOfMindAgent),
        "loaded_name": loaded.name
    }


async def society_of_mind_agent_empty_messages(runtime: Optional[AgentRuntime]):
    agent1 = AssistantAgent("assistant1", model_client=model_client,
                            system_message="You are a helpful assistant.")
    agent2 = AssistantAgent("assistant2", model_client=model_client2,
                            system_message="You are a helpful assistant.")
    team = RoundRobinGroupChat(
        [agent1, agent2], MaxMessageTermination(3), runtime=runtime)
    society = SocietyOfMindAgent(
        "society_of_mind", team=team, model_client=model_client)

    response = await society.run()
    return make_serializable(response)


async def society_of_mind_agent_no_response(runtime: Optional[AgentRuntime]):
    agent1 = AssistantAgent("assistant1", model_client=model_client,
                            system_message="You are a helpful assistant.")
    agent2 = AssistantAgent("assistant2", model_client=model_client2,
                            system_message="You are a helpful assistant.")
    team = RoundRobinGroupChat(
        [agent1, agent2], MaxMessageTermination(1), runtime=runtime)
    society = SocietyOfMindAgent(
        "society_of_mind", team=team, model_client=model_client)

    response = await society.run(task="Count to 10.")
    return {
        "response": make_serializable(response),
        "final_text": response.messages[1].to_text() if len(response.messages) > 1 else None
    }


async def main():
    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    try:
        print("\n=== Multiple Rounds Test ===")
        result = await society_of_mind_agent_multiple_rounds(runtime)
        logger.success(format_json(result))

        print("\n=== Basic Function ===")
        result = await society_of_mind_agent_basic(runtime)
        logger.success(format_json(result))

        print("\n=== Empty Message Task ===")
        result = await society_of_mind_agent_empty_messages(runtime)
        logger.success(format_json(result))

        print("\n=== No Response Test ===")
        result = await society_of_mind_agent_no_response(runtime)
        logger.success(format_json(result))

    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
