import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Solving Multiple Tasks in a Sequence of Async Chats

This notebook showcases how to use the new chat interface of conversational agents in AutoGen: a_initiate_chats, to conduct a series of tasks. Similar to "notebook/agentchat_microsoft_fabric.ipynb", this new interface allows one to pass multiple tasks and their corresponding dedicated agents and execute concurrently. Depending on the prerequisite task(s), the tasks will be solved concurrently, with the summaries from prerequisite task(s) provided to subsequent tasks as context, if the `summary_method` argument is specified.

\:\:\:info Requirements

Install `pyautogen`:
```bash
pip install pyautogen
```

For more information, please refer to the [installation guide](/docs/installation/).

\:\:\:
"""
logger.info("# Solving Multiple Tasks in a Sequence of Async Chats")


config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

"""
\:\:\:tip

Learn more about the various ways to configure LLM endpoints [here](/docs/topics/llm_configuration).

\:\:\:

### Example Tasks
Below are four example tasks, with each task being a string of text describing the request. The completion of later tasks requires or benefits from the results of prerequisite tasks.
"""
logger.info("### Example Tasks")

financial_tasks = [
    """What are the current stock prices of NVDA and TESLA, and how is the performance over the past month in terms of percentage change?""",
    """Investigate possible reasons of the stock performance.""",
    """Plot a graph comparing the stock prices over the past month.""",
]

writing_tasks = ["""Develop an engaging blog post using any information provided."""]

"""
### Scenario 1: Solve the tasks with a series of chats

The `initiate_chats` interface can take a list of dictionaries as inputs. Each dictionary preserves the following fields: 
- `message`: is a string of text (typically a message containing the task);
- `recipient`: a conversable agent dedicated for the task;
- `summary_method`: A string specifying the method to get a summary from the chat. Currently supported choices include `last_msg`, which takes the last message from the chat history as the summary, and `reflection_with_llm`, which uses an LLM call to reflect on the chat history and summarize a takeaway;
- `summary_prompt`: A string specifying how to instruct an LLM-backed agent (either the recipient or the sender in the chat) to reflect on the chat history and derive a summary. If not otherwise specified, a default prompt will be used when `summary_method` is `reflection_with_llm`.
"Summarize the takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
- `carryover`: A string or a list of string to specify additional context to be used in the chat. With `initiate_chats`, summary from previous chats will be added as carryover. They will be appended after the carryover provided by the user.
"""
logger.info("### Scenario 1: Solve the tasks with a series of chats")

financial_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)
research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
)
writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful and engaging articles.
        You transform complex concepts into compelling narratives.
        Reply "TERMINATE" in the end when everything is done.
        """,
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

async def run_async_code_990c199f():
    chat_results = await user.a_initiate_chats(  # noqa: F704
    return chat_results
chat_results = asyncio.run(run_async_code_990c199f())
logger.success(format_json(chat_results))
    [
        {
            "chat_id": 1,
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 2,
            "prerequisites": [1],
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 3,
            "prerequisites": [1],
            "recipient": financial_assistant,
            "message": financial_tasks[2],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {"chat_id": 4, "prerequisites": [1, 2, 3], "recipient": writer, "silent": False, "message": writing_tasks[0]},
    ]
)

"""
#### Check chat results
The `initiate_chat` method returns a `ChatResult` object, which is a dataclass object storing information about the chat. Currently, it includes the following attributes:

- `chat_history`: a list of chat history.
- `summary`: a string of chat summary. A summary is only available if a summary_method is provided when initiating the chat.
- `cost`: a tuple of (total_cost, total_actual_cost), where total_cost is a dictionary of cost information, and total_actual_cost is a dictionary of information on the actual incurred cost with cache.
- `human_input`: a list of strings of human inputs solicited during the chat. (Note that since we are setting `human_input_mode` to `NEVER` in this notebook, this list is always empty.)
"""
logger.info("#### Check chat results")

for i, chat_res in chat_results.items():
    logger.debug(f"*****{i}th chat*******:")
    logger.debug(chat_res.summary)
    logger.debug("Human input in the middle:", chat_res.human_input)
    logger.debug("Conversation cost: ", chat_res.cost)
    logger.debug("\n\n")

"""
### Scenario 2: With human inputs revising tasks in the middle

Since AutoGen agents support soliciting human inputs during a chat if `human_input_mode` is specificed properly, the actual task might be revised in the middle of a chat. 

The example below showcases that even if a task is revised in the middle (for the first task, the human user requests to get Microsoft's stock price information as well, in addition to NVDA and TSLA), the `reflection_with_llm`` summary method can still capture it, as it reflects on the whole conversation instead of just the original request.
"""
logger.info("### Scenario 2: With human inputs revising tasks in the middle")

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",  # ask human for input at each step
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

async def run_async_code_990c199f():
    chat_results = await user.a_initiate_chats(  # noqa: F704
    return chat_results
chat_results = asyncio.run(run_async_code_990c199f())
logger.success(format_json(chat_results))
    [
        {
            "chat_id": 1,
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 2,
            "prerequisites": [1],
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "silent": True,
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 3,
            "prerequisites": [1],
            "recipient": financial_assistant,
            "message": financial_tasks[2],
            "silent": False,
            "summary_method": "reflection_with_llm",
        },
        {"chat_id": 4, "prerequisites": [1, 2, 3], "recipient": writer, "message": writing_tasks[0]},
    ]
)

"""
#### Check chat results
"""
logger.info("#### Check chat results")

for i, chat_res in chat_results.items():
    logger.debug(f"*****{i}th chat*******:")
    logger.debug(chat_res.summary)
    logger.debug("Human input in the middle:", chat_res.human_input)
    logger.debug("Conversation cost: ", chat_res.cost)
    logger.debug("\n\n")

"""
### Scenario 3: Solve the tasks with a series of chats involving group chat
"""
logger.info("### Scenario 3: Solve the tasks with a series of chats involving group chat")

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a professional writer, known for
    your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    Reply "TERMINATE" in the end when everything is done.
    """,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.
    Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config={"config_list": config_list},
)

groupchat_1 = autogen.GroupChat(agents=[user_proxy, research_assistant, critic], messages=[], max_round=50)

groupchat_2 = autogen.GroupChat(agents=[user_proxy, writer, critic], messages=[], max_round=50)

manager_1 = autogen.GroupChatManager(
    groupchat=groupchat_1,
    name="Research_manager",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)
manager_2 = autogen.GroupChatManager(
    groupchat=groupchat_2,
    name="Writing_manager",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)
async def run_async_code_b1de1d29():
    await user.a_initiate_chats(  # noqa: F704
    return 
 = asyncio.run(run_async_code_b1de1d29())
logger.success(format_json())
    [
        {"chat_id": 1, "recipient": research_assistant, "message": financial_tasks[0], "summary_method": "last_msg"},
        {
            "chat_id": 2,
            "prerequisites": [1],
            "recipient": manager_1,
            "message": financial_tasks[1],
            "summary_method": "reflection_with_llm",
        },
        {
            "chat_id": 3,
            "prerequisites": [1],
            "recipient": manager_1,
            "message": financial_tasks[2],
            "summary_method": "reflection_with_llm",
        },
        {"chat_id": 4, "prerequisites": [1, 2, 3], "recipient": manager_2, "message": writing_tasks[0]},
    ]
)

logger.info("\n\n[DONE]", bright=True)