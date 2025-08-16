from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, register_function
from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from tavily import TavilyClient
from typing import Annotated
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# ReAct

AutoGen supports different LLM prompting and reasoning strategies, such as ReAct, Reflection/Self-Critique, and more. This page demonstrates how to realize ReAct ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) with AutoGen.

First make sure required packages are installed.
"""
logger.info("# ReAct")

# ! pip install "pyautogen>=0.2.18" "tavily-python"

"""
Import the relevant modules and configure the LLM.
See [LLM Configuration](/docs/topics/llm_configuration) for how to configure LLMs.
"""
logger.info("Import the relevant modules and configure the LLM.")




config_list = [
#     {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]},
#     {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
]

"""
### Define and add tools/actions you want LLM agent(s) to use

We use [Tavily](https://docs.tavily.com/docs/tavily-api/python-sdk) as a tool for searching the web.
"""
logger.info("### Define and add tools/actions you want LLM agent(s) to use")

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

"""
### Construct your ReAct prompt
Here we are constructing a general ReAct prompt and a custom message function `react_prompt_message` based on the ReAct prompt.
"""
logger.info("### Construct your ReAct prompt")

ReAct_prompt = """
Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
"""



def react_prompt_message(sender, recipient, context):
    return ReAct_prompt.format(input=context["question"])

"""
### Construct agents and initiate agent chats
"""
logger.info("### Construct agents and initiate agent chats")

os.makedirs("coding", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

assistant = AssistantAgent(
    name="Assistant",
    system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

register_function(
    search_tool,
    caller=assistant,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

with Cache.disk(cache_seed=43) as cache:
    user_proxy.initiate_chat(
        assistant,
        message=react_prompt_message,
        question="What is the result of super bowl 2024?",
        cache=cache,
    )

"""
### ReAct with memory module enabled via teachability
"""
logger.info("### ReAct with memory module enabled via teachability")

teachability = teachability.Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

teachability.add_to_agent(assistant)

with Cache.disk(cache_seed=44) as cache:
    user_proxy.initiate_chat(
        assistant,
        message=react_prompt_message,
        question="What is the result of super bowl 2024?",
        cache=cache,
    )

"""
### Let's now ask the same question again to see if the assistant has remembered this.
"""
logger.info("### Let's now ask the same question again to see if the assistant has remembered this.")

with Cache.disk(cache_seed=110) as cache:
    user_proxy.initiate_chat(
        assistant,
        message=react_prompt_message,
        question="What is the result of super bowl 2024?",
        max_turns=1,
        cache=cache,
    )

logger.info("\n\n[DONE]", bright=True)