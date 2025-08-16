import autogen

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Solving Complex Tasks with A Sequence of Nested Chats

This notebook shows how you can leverage **nested chats** to solve complex task with AutoGen. Nested chats is a sequence of chats created by a receiver agent after receiving a message from a sender agent and finished before the receiver agent replies to this message. Nested chats allow AutoGen agents to use other agents as their inner monologue to accomplish tasks. This abstraction is powerful as it allows you to compose agents in rich ways. This notebook shows how you can nest a pretty complex sequence of chats among _inner_ agents inside an _outer_ agent.

\:\:\:info Requirements

Install `autogen`:
```bash
pip install autogen
```

For more information, please refer to the [installation guide](/docs/installation/).

\:\:\:
"""
logger.info("# Solving Complex Tasks with A Sequence of Nested Chats")


config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

"""
\:\:\:tip

Learn more about the various ways to configure LLM endpoints [here](/docs/topics/llm_configuration).

\:\:\:

### Example Task

Suppose we want the agents to complete the following sequence of tasks:
"""
logger.info("### Example Task")

tasks = [
    """On which days in 2024 was Microsoft Stock higher than $400? Comment on the stock performance.""",
    """Make a pleasant joke about it.""",
]

"""
Since the first task could be complex to solve, lets construct new agents that can serve as an inner monologue.

### Step 1. Define Agents

#### A Group Chat for Inner Monologue
Below, we construct a group chat manager which manages an `inner_assistant` agent and an `inner_code_interpreter` agent. 
Later we will use this group chat inside another agent.
"""
logger.info("### Step 1. Define Agents")

inner_assistant = autogen.AssistantAgent(
    "Inner-assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

inner_code_interpreter = autogen.UserProxyAgent(
    "Inner-code-interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

groupchat = autogen.GroupChat(
    agents=[inner_assistant, inner_code_interpreter],
    messages=[],
    speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
    allow_repeat_speaker=False,
    max_round=8,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
)

"""
#### Inner- and Outer-Level Individual Agents

Now we will construct a number of individual agents that will assume role of outer and inner agents.
"""
logger.info("#### Inner- and Outer-Level Individual Agents")

assistant_1 = autogen.AssistantAgent(
    name="Assistant_1",
    llm_config={"config_list": config_list},
)

assistant_2 = autogen.AssistantAgent(
    name="Assistant_2",
    llm_config={"config_list": config_list},
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a professional writer, known for
    your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    """,
)

reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a compliance reviewer, known for your thoroughness and commitment to standards.
    Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
    all materials align with required guidelines.
    You must review carefully, identify potential issues, and maintain the integrity of the organization.
    Your role demands fairness, a deep understanding of regulations, and a focus on protecting against
    harm while upholding a culture of responsibility.
    """,
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

"""
### Step 2: Orchestrate Nested Chats to Solve Tasks

#### Outer Level
In the following code block, at the outer level, we have communication between:

- `user` - `assistant_1` for solving the first task, i.e., `tasks[0]`.
- `user` - `assistant_2` for solving the second task, i.e., `tasks[1]`.

#### Inner Level (Nested Chats)
Since the first task is quite complicated, we created a sequence of _nested chats_ as the inner monologue of Assistant_1.

1. `assistant_1` - `manager`: This chat intends to delegate the task received by Assistant_1 to the Manager to solve.

2. `assistant_1` - `writer`: This chat takes the output from Nested Chat 1, i.e., Assistant_1 vs. Manager, and lets the Writer polish the content to make an engaging and nicely formatted blog post, which is realized through the writing_message function.

3. `assistant_1` - `reviewer`: This chat takes the output from Nested Chat 2 and intends to let the Reviewer agent review the content from Nested Chat 2.

4. `assistant_1` - `writer`: This chat takes the output from previous nested chats and intends to let the Writer agent finalize a blog post.

The sequence of nested chats can be realized with the `register_nested_chats` function, which allows one to register one or a sequence of chats to a particular agent (in this example, the `assistant_1` agent).

Information about the sequence of chats can be specified in the `chat_queue` argument of the `register_nested_chats` function. The following fields are especially useful:
- `recipient` (required) specifies the nested agent;
- `message` specifies what message to send to the nested recipient agent. In a sequence of nested chats, if the `message` field is not specified, we will use the last message the registering agent received as the initial message in the first chat and will skip any subsequent chat in the queue that does not have the `message` field. You can either provide a string or define a callable that returns a string.
- `summary_method` decides what to get out of the nested chat. You can either select from existing options including `"last_msg"` and `"reflection_with_llm"`, or or define your own way on what to get from the nested chat with a Callable.
- `max_turns` determines how many turns of conversation to have between the concerned agent pairs.
"""
logger.info("### Step 2: Orchestrate Nested Chats to Solve Tasks")

def writing_message(recipient, messages, sender, config):
    return f"Polish the content to make an engaging and nicely formatted blog post. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


nested_chat_queue = [
    {"recipient": manager, "summary_method": "reflection_with_llm"},
    {"recipient": writer, "message": writing_message, "summary_method": "last_msg", "max_turns": 1},
    {"recipient": reviewer, "message": "Review the content provided.", "summary_method": "last_msg", "max_turns": 1},
    {"recipient": writer, "message": writing_message, "summary_method": "last_msg", "max_turns": 1},
]
assistant_1.register_nested_chats(
    nested_chat_queue,
    trigger=user,
)

res = user.initiate_chats(
    [
        {"recipient": assistant_1, "message": tasks[0], "max_turns": 1, "summary_method": "last_msg"},
        {"recipient": assistant_2, "message": tasks[1]},
    ]
)

logger.info("\n\n[DONE]", bright=True)