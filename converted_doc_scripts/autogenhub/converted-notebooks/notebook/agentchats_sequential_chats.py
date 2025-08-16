from jet.logger import CustomLogger
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Solving Multiple Tasks in a Sequence of Chats with Different Conversable Agent Pairs

This notebook showcases how to use the new chat interface `autogen.initiate_chats` to solve a set of tasks with a sequence of chats. 

\:\:\:info Requirements

Install `pyautogen`:
```bash
pip install pyautogen
```

For more information, please refer to the [installation guide](/docs/installation/).

\:\:\:
"""
logger.info("# Solving Multiple Tasks in a Sequence of Chats with Different Conversable Agent Pairs")


config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

"""
\:\:\:tip

Learn more about the various ways to configure LLM endpoints [here](/docs/topics/llm_configuration).

\:\:\:

### Example Tasks
Below are three example tasks, with each task being a string of text describing the request. The completion of later tasks requires or benefits from the results of previous tasks.
"""
logger.info("### Example Tasks")

financial_tasks = [
    """What are the current stock prices of NVDA and TESLA, and how is the performance over the past month in terms of percentage change?""",
    """Investigate possible reasons of the stock performance leveraging market news.""",
]

writing_tasks = ["""Develop an engaging blog post using any information provided."""]

"""
### Example 1: Solve tasks with a series of chats

The `autogen.initiate_chats` interface can take a list of dictionaries as inputs. Each dictionary preserves the following fields: 

- `sender`: a conversable agent as the sender;
- `recipient`: a conversable agent as the recipient;
- `message`: is a string of text (typically a message containing the task) or a callable;
- `summary_method`: A string or a callable to get a summary from the chat. Currently supported choices include `last_msg`, which takes the last message from the chat history as the summary, and `reflection_with_llm`, which uses an LLM call to reflect on the chat history and summarize a takeaway;
"""
logger.info("### Example 1: Solve tasks with a series of chats")

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

user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS",  # ask human for input at each step
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


chat_results = autogen.initiate_chats(
    [
        {
            "sender": user_proxy_auto,
            "recipient": financial_assistant,
            "message": financial_tasks[0],
            "clear_history": True,
            "silent": False,
            "summary_method": "last_msg",
        },
        {
            "sender": user_proxy_auto,
            "recipient": research_assistant,
            "message": financial_tasks[1],
            "max_turns": 2,  # max number of turns for the conversation (added for demo purposes, generally not necessarily needed)
            "summary_method": "reflection_with_llm",
        },
        {
            "sender": user_proxy,
            "recipient": writer,
            "message": writing_tasks[0],
            "carryover": "I want to include a figure or a table of data in the blogpost.",  # additional carryover to include to the conversation (added for demo purposes, generally not necessarily needed)
        },
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

for i, chat_res in enumerate(chat_results):
    logger.debug(f"*****{i}th chat*******:")
    logger.debug(chat_res.summary)
    logger.debug("Human input in the middle:", chat_res.human_input)
    logger.debug("Conversation cost: ", chat_res.cost)
    if i == 1:
        assert (
            len(chat_res.chat_history) == 4
        ), f"The chat history should contain at most 4 messages because max_turns is set to 2 in the {i}-th chat."
    logger.debug("\n\n")

"""
### Example 2: Solve a Sequence of Tasks involving User Defined Message

In this example, say I have two tasks. One resarch task and a one writing task. The writing task needs data from research task. In this example, we direct read data from a file as part of the message.
"""
logger.info("### Example 2: Solve a Sequence of Tasks involving User Defined Message")

research_task = """What are daily stock prices of NVDA and TESLA in the past month. Save the results in a .md file named 'stock_prices.md'."""


def my_writing_task(sender, recipient, context):
    carryover = context.get("carryover", "")
    if isinstance(carryover, list):
        carryover = carryover[-1]

    try:
        filename = context.get("work_dir", "") + "/stock_prices.md"
        with open(filename, "r") as file:
            data = file.read()
    except Exception as e:
        data = f"An error occurred while reading the file: {e}"

    return (
        """Develop an engaging blog post using any information provided. """
        + "\nContext:\n"
        + carryover
        + "\nData:"
        + data
    )

researcher = autogen.AssistantAgent(
    name="Financial_researcher",
    llm_config=llm_config,
)
writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful and engaging articles.
        You transform complex concepts into compelling narratives.
        Reply "TERMINATE" in the end when everything is done.
        """,
)

user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

chat_results = autogen.initiate_chats(
    [
        {
            "sender": user_proxy_auto,
            "recipient": researcher,
            "message": research_task,
            "clear_history": True,
            "silent": False,
            "summary_method": "last_msg",
        },
        {
            "sender": user_proxy_auto,
            "recipient": writer,
            "message": my_writing_task,
            "max_turns": 2,  # max number of turns for the conversation (added for demo purposes, generally not necessarily needed)
            "summary_method": "reflection_with_llm",
            "work_dir": "tasks",
        },
    ]
)

logger.info("\n\n[DONE]", bright=True)