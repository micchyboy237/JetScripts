from jet.logger import CustomLogger
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Customize Speaker Selection

```{=mdx}
![group_chat](../../../blog/2024-02-29-StateFlow/img/sf_example_1.png)
```

In GroupChat, we can customize the speaker selection by passing a function to the `GroupChat` object. With this function, you can build a more **deterministic** agent workflow. We recommend following a **StateFlow** pattern when crafting this function. Please refer to the [StateFlow blog](/blog/2024/02/29/StateFlow) for more details.


## An example research workflow
We provide a simple example to build a StateFlow model for research with customized speaker selection.

We first define the following agents:

- Initializer: Start the workflow by sending a task.
- Coder: Retrieve papers from the internet by writing code.
- Executor: Execute the code.
- Scientist: Read the papers and write a summary.

In the figure above, we define a simple workflow for research with 4 states: *Init*, *Retrieve*, *Research*, and *End*. Within each state, we will call different agents to perform the tasks.

- *Init*: We use the initializer to start the workflow.
- *Retrieve*: We will first call the coder to write code and then call the executor to execute the code.
- *Research*: We will call the scientist to read the papers and write a summary.
- *End*: We will end the workflow.

## Create your speaker selection function

Below is a skeleton of the speaker selection function. Fill in the function to define the speaker selection logic.

```python
def custom_speaker_selection_func(
    last_speaker: Agent, 
    groupchat: GroupChat
) -> Union[Agent, Literal['auto', 'manual', 'random' 'round_robin'], None]:

    """
logger.info("# Customize Speaker Selection")Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Parameters:
        - last_speaker: Agent
            The last speaker in the group chat.
        - groupchat: GroupChat
            The GroupChat object
    Return:
        Return one of the following:
        1. an `Agent` class, it must be one of the agents in the group chat.
        2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        3. None, which indicates the chat should be terminated.
    """
    pass

groupchat = autogen.GroupChat(
    speaker_selection_method=custom_speaker_selection_func,
    ...,
)
```
The last speaker and the groupchat object are passed to the function. 
Commonly used variables from groupchat are `groupchat.messages` and `groupchat.agents`, which is the message history and the agents in the group chat respectively. You can access other attributes of the groupchat, such as `groupchat.allowed_speaker_transitions_dict` for pre-defined `allowed_speaker_transitions_dict`.
"""
logger.info("pass")



config_list = [
    {
        "model": "gpt-4-0125-preview",
#         "api_key": os.environ["OPENAI_API_KEY"],
    }
]

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

initializer = autogen.UserProxyAgent(
    name="Init",
)

coder = autogen.AssistantAgent(
    name="Retrieve_Action_1",
    llm_config=gpt4_config,
    system_message="""You are the Coder. Given a topic, write code to retrieve related papers from the arXiv API, print their title, authors, abstract, and link.
You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
executor = autogen.UserProxyAgent(
    name="Retrieve_Action_2",
    system_message="Executor. Execute the code written by the Coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)
scientist = autogen.AssistantAgent(
    name="Research_Action_1",
    llm_config=gpt4_config,
    system_message="""You are the Scientist. Please categorize papers after seeing their abstracts printed and create a markdown table with Domain, Title, Authors, Summary and Link""",
)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        return coder
    elif last_speaker is coder:
        return executor
    elif last_speaker is executor:
        if messages[-1]["content"] == "exitcode: 1":
            return coder
        else:
            return scientist
    elif last_speaker == "Scientist":
        return None


groupchat = autogen.GroupChat(
    agents=[initializer, coder, executor, scientist],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

initializer.initiate_chat(
    manager, message="Topic: LLM applications papers from last week. Requirement: 5 - 10 papers from different domains."
)

logger.info("\n\n[DONE]", bright=True)