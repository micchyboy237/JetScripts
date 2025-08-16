from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LLM Reflection

AutoGen supports different LLM prompting and reasoning strategies, such as ReAct, Reflection/Self-Critique, and more. This notebook demonstrates how to realize general LLM reflection with AutoGen. Reflection is a general prompting strategy which involves having LLMs analyze their own outputs, behaviors, knowledge, or reasoning processes.

This example leverages a generic interface [nested chats](/docs/tutorial/conversation-patterns#nested-chats) to implement the reflection using multi-agents.

First make sure the `pyautogen` package is installed.
"""
logger.info("# LLM Reflection")

# ! pip install "pyautogen>=0.2.18"

"""
Import the relevant modules and configure the LLM.
See [LLM Configuration](/docs/topics/llm_configuration) for how to configure LLMs.
"""
logger.info("Import the relevant modules and configure the LLM.")



config_list = [
#     {"model": "gpt-4-1106-preview", "api_key": os.environ["OPENAI_API_KEY"]},
#     {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
]

"""
## Construct Agents
Now we create three agents, including `user_proxy` as a user proxy, `writing_assistant` for generating solutions (based on the initial request or critique), and `reflection_assistant` for reflecting and providing critique.
"""
logger.info("## Construct Agents")

os.makedirs("coding", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

writing_assistant = AssistantAgent(
    name="writing_assistant",
    system_message="You are an writing assistant tasked to write engaging blogpost. You try generate the best blogpost possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

reflection_assistant = AssistantAgent(
    name="reflection_assistant",
    system_message="Generate critique and recommendations on the writing. Provide detailed recommendations, including requests for length, depth, style, etc..",
    llm_config={"config_list": config_list, "cache_seed": None},
)

"""
## Construct Agent Chats with `reflection_assistant` being a Nested Agent for Reflection
"""
logger.info("## Construct Agent Chats with `reflection_assistant` being a Nested Agent for Reflection")

def reflection_message(recipient, messages, sender, config):
    logger.debug("Reflecting...")
    return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


nested_chat_queue = [
    {
        "recipient": reflection_assistant,
        "message": reflection_message,
        "max_turns": 1,
    },
]
user_proxy.register_nested_chats(
    nested_chat_queue,
    trigger=writing_assistant,
)

with Cache.disk(cache_seed=42) as cache:
    user_proxy.initiate_chat(
        writing_assistant,
        message="Write an engaging blogpost on the recent updates in AI. "
        "The blogpost should be engaging and understandable for general audience. "
        "Should have more than 3 paragraphes but no longer than 1000 words.",
        max_turns=2,
        cache=cache,
    )

logger.info("\n\n[DONE]", bright=True)