import os
import shutil
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main() -> None:
    model_client = MLXAutogenChatLLMAdapter(
        model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", seed=42)

    writer = AssistantAgent(
        name="writer",
        description="A writer.",
        system_message="You are a writer.",
        model_client=model_client,
    )

    critic = AssistantAgent(
        name="critic",
        description="A critic.",
        system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
        model_client=model_client,
    )

    # The termination condition is a text termination, which will cause the chat to terminate when the text "APPROVE" is received.
    termination = TextMentionTermination("APPROVE")

    # The group chat will alternate between the writer and the critic.
    group_chat = RoundRobinGroupChat(
        [writer, critic], termination_condition=termination, max_turns=12)

    # `run_stream` returns an async generator to stream the intermediate messages.
    stream = group_chat.run_stream(
        task="Write a short story about a robot that discovers it has feelings.")
    # `Console` is a simple UI to display the stream.
    await Console(stream)
    # Close the connection to the model client.
    await model_client.close()

asyncio.run(main())
