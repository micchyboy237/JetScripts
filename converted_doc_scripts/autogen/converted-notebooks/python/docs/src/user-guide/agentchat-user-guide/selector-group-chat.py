from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchAPIWrapper, DuckDuckGoSearchRun
import asyncio
import sys
import uuid
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.logger import CustomLogger
from typing import List, Sequence
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_client = OllamaChatCompletionClient(
    model="llama3.2", host="http://jethros-macbook-air.local:11434")

"""
# Selector Group Chat

{py:class}`~autogen_agentchat.teams.SelectorGroupChat` implements a team where participants take turns broadcasting messages to all other members. A generative model (e.g., an LLM) selects the next speaker based on the shared context, enabling dynamic, context-aware collaboration.

Key features include:

- Model-based speaker selection
- Configurable participant roles and descriptions
- Prevention of consecutive turns by the same speaker (optional)
- Customizable selection prompting
- Customizable selection function to override the default model-based selection
- Customizable candidate function to narrow-down the set of agents for selection using model

```{note}
{py:class}`~autogen_agentchat.teams.SelectorGroupChat` is a high-level API. For more control and customization, refer to the [Group Chat Pattern](../core-user-guide/design-patterns/group-chat.ipynb) in the Core API documentation to implement your own group chat logic.
```

## How Does it Work?

{py:class}`~autogen_agentchat.teams.SelectorGroupChat` is a group chat similar to {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`,
but with a model-based next speaker selection mechanism.
When the team receives a task through {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` or {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream`,
the following steps are executed:

1. The team analyzes the current conversation context, including the conversation history and participants' {py:attr}`~autogen_agentchat.base.ChatAgent.name` and {py:attr}`~autogen_agentchat.base.ChatAgent.description` attributes, to determine the next speaker using a model. By default, the team will not select the same speak consecutively unless it is the only agent available. This can be changed by setting `allow_repeated_speaker=True`. You can also override the model by providing a custom selection function.
2. The team prompts the selected speaker agent to provide a response, which is then **broadcasted** to all other participants.
3. The termination condition is checked to determine if the conversation should end, if not, the process repeats from step 1.
4. When the conversation ends, the team returns the {py:class}`~autogen_agentchat.base.TaskResult` containing the conversation history from this task.

Once the team finishes the task, the conversation context is kept within the team and all participants, so the next task can continue from the previous conversation context.
You can reset the conversation context by calling {py:meth}`~autogen_agentchat.teams.BaseGroupChat.reset`.

In this section, we will demonstrate how to use {py:class}`~autogen_agentchat.teams.SelectorGroupChat` with a simple example for a web search and data analysis task.

## Example: Web Search/Analysis
"""
logger.info("# Selector Group Chat")


"""
### Agents

![Selector Group Chat](selector-group-chat.svg)

This system uses three specialized agents:

- **Planning Agent**: The strategic coordinator that breaks down complex tasks into manageable subtasks. 
- **Web Search Agent**: An information retrieval specialist that interfaces with the `search_web_tool`.
- **Data Analyst Agent**: An agent specialist in performing calculations equipped with `percentage_change_tool`.

The tools `search_web_tool` and `percentage_change_tool` are external tools that the agents can use to perform their tasks.
"""
logger.info("### Agents")


def search_web_tool(query: str) -> str:
    api_wrapper = DuckDuckGoSearchAPIWrapper(
        region="wt-wt",
        safesearch="moderate",
        time="y",
        max_results=10,
        source="text"
    )
    search_tool = DuckDuckGoSearchRun(api_wrapper=api_wrapper)
    result = search_tool._run(query)
    return result


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


"""
Let's create the specialized agents using the {py:class}`~autogen_agentchat.agents.AssistantAgent` class.
It is important to note that the agents' {py:attr}`~autogen_agentchat.base.ChatAgent.name` and {py:attr}`~autogen_agentchat.base.ChatAgent.description` attributes are used by the model to determine the next speaker,
so it is recommended to provide meaningful names and descriptions.
"""
logger.info(
    "Let's create the specialized agents using the {py:class}`~autogen_agentchat.agents.AssistantAgent` class.")

conversation_id = str(uuid.uuid4())

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        WebSearchAgent: Searches for information
        DataAnalystAgent: Performs calculations

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    If you have not seen the data, ask for it.
    """,
)

"""
```{note}
By default, {py:class}`~autogen_agentchat.agents.AssistantAgent` returns the
tool output as the response. If your tool does not return a well-formed
string in natural language format, you may want to add a reflection step
within the agent by setting `reflect_on_tool_use=True` when creating the agent.
This will allow the agent to reflect on the tool output and provide a natural
language response.
```

### Workflow

1. The task is received by the {py:class}`~autogen_agentchat.teams.SelectorGroupChat` which, based on agent descriptions, selects the most appropriate agent to handle the initial task (typically the Planning Agent).

2. The **Planning Agent** analyzes the task and breaks it down into subtasks, assigning each to the most appropriate agent using the format:
   `<agent> : <task>`

3. Based on the conversation context and agent descriptions, the {py:class}`~autogen_agent.teams.SelectorGroupChat` manager dynamically selects the next agent to handle their assigned subtask.

4. The **Web Search Agent** performs searches one at a time, storing results in the shared conversation history.

5. The **Data Analyst** processes the gathered information using available calculation tools when selected.

6. The workflow continues with agents being dynamically selected until either:
   - The Planning Agent determines all subtasks are complete and sends "TERMINATE"
   - An alternative termination condition is met (e.g., a maximum number of messages)

When defining your agents, make sure to include a helpful {py:attr}`~autogen_agentchat.base.ChatAgent.description` since this is used to decide which agent to select next.

### Termination Conditions

Let's use two termination conditions:
{py:class}`~autogen_agentchat.conditions.TextMentionTermination` to end the conversation when the Planning Agent sends "TERMINATE",
and {py:class}`~autogen_agentchat.conditions.MaxMessageTermination` to limit the conversation to 25 messages to avoid infinite loop.
"""
logger.info("### Workflow")

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

"""
### Selector Prompt

{py:class}`~autogen_agentchat.teams.SelectorGroupChat` uses a model to select
the next speaker based on the conversation context.
We will use a custom selector prompt to properly align with the workflow.
"""
logger.info("### Selector Prompt")

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

"""
The string variables available in the selector prompt are:
- `{participants}`: The names of candidates for selection. The format is `["<name1>", "<name2>", ...]`.
- `{roles}`: A newline-separated list of names and descriptions of the candidate agents. The format for each line is: `"<name> : <description>"`.
- `{history}`: The conversation history formatted as a double newline separated of names and message content. The format for each message is: `"<name> : <message content>"`.

```{tip}
Try not to overload the model with too much instruction in the selector prompt.

What is too much? It depends on the capabilities of the model you are using.
For GPT-4o and equivalents, you can use a selector prompt with a condition for when each speaker should be selected.
For smaller models such as Phi-4, you should keep the selector prompt as simple as possible
such as the one used in this example.

Generally, if you find yourself writing multiple conditions for each agent,
it is a sign that you should consider using a custom selection function,
or breaking down the task into smaller, sequential tasks to be handled by
separate agents or teams.
```

### Running the Team

Let's create the team with the agents, termination conditions, and custom selector prompt.
"""
logger.info("### Running the Team")

team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    name="GroupChatManager",
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    # Allow an agent to speak multiple turns in a row.
    allow_repeated_speaker=True,
)

"""
Now we run the team with a task to find information about an NBA player.
"""
logger.info(
    "Now we run the team with a task to find information about an NBA player.")

task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"


async def run_async_code_ac2575cc():
    await Console(team.run_stream(task=task))
asyncio.run(run_async_code_ac2575cc())

sys.exit()

"""
As we can see, after the Web Search Agent conducts the necessary searches and the Data Analyst Agent completes the necessary calculations, we find that Dwayne Wade was the Miami Heat player with the highest points in the 2006-2007 season, and the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons is 85.98%!

## Custom Selector Function

Often times we want better control over the selection process.
To this end, we can set the `selector_func` argument with a custom selector function to override the default model-based selection.
This allows us to implement more complex selection logic and state-based transitions.

For instance, we want the Planning Agent to speak immediately after any specialized agent to check the progress.

```{note}
Returning `None` from the custom selector function will use the default model-based selection.
``` 

```{note}
Custom selector functions are not [serialized](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/serialize-components.html) when `.dump_component()` is called on the SelectorGroupChat team . If you need to serialize team configurations with custom selector functions, consider implementing custom workflows and serialization logic.
```
"""
logger.info("## Custom Selector Function")


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name:
        return planning_agent.name
    return None


async def run_async_code_a5b70700():
    await team.reset()
asyncio.run(run_async_code_a5b70700())
team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
    selector_func=selector_func,
)


async def run_async_code_ac2575cc():
    await Console(team.run_stream(task=task))
asyncio.run(run_async_code_ac2575cc())

"""
You can see from the conversation log that the Planning Agent always speaks immediately after the specialized agents.

```{tip}
Each participant agent only makes one step (executing tools, generating a response, etc.)
on each turn. 
If you want an {py:class}`~autogen_agentchat.agents.AssistantAgent` to repeat
until it stop returning a {py:class}`~autogen_agentchat.messages.ToolCallSummaryMessage`
when it has finished running all the tools it needs to run, you can do so by
checking the last message and returning the agent if it is a
{py:class}`~autogen_agentchat.messages.ToolCallSummaryMessage`.
```

## Custom Candidate Function

One more possible requirement might be to automatically select the next speaker from a filtered list of agents.
For this, we can set `candidate_func` parameter with a custom candidate function to filter down the list of potential agents for speaker selection for each turn of groupchat.

This allow us to restrict speaker selection to a specific set of agents after a given agent.


```{note}
The `candidate_func` is only valid if `selector_func` is not set.
Returning `None` or an empty list `[]` from the custom candidate function will raise a `ValueError`.
```
"""
logger.info("## Custom Candidate Function")


def candidate_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
    if messages[-1].source == "user":
        return [planning_agent.name]

    last_message = messages[-1]
    if last_message.source == planning_agent.name:
        participants = []
        if web_search_agent.name in last_message.to_text():
            participants.append(web_search_agent.name)
        if data_analyst_agent.name in last_message.to_text():
            participants.append(data_analyst_agent.name)
        if participants:
            # SelectorGroupChat will select from the remaining two agents.
            return participants

    previous_set_of_agents = set(message.source for message in messages)
    if web_search_agent.name in previous_set_of_agents and data_analyst_agent.name in previous_set_of_agents:
        return [planning_agent.name]

    return [planning_agent.name, web_search_agent.name, data_analyst_agent.name]


async def run_async_code_a5b70700():
    await team.reset()
asyncio.run(run_async_code_a5b70700())
team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    candidate_func=candidate_func,
)


async def run_async_code_ac2575cc():
    await Console(team.run_stream(task=task))
asyncio.run(run_async_code_ac2575cc())

"""
You can see from the conversation log that the Planning Agent returns to conversation once the Web Search Agent and Data Analyst Agent took their turns and it finds that the task was not finished as expected so it called the WebSearchAgent again to get rebound values and then called DataAnalysetAgent to get the percentage change.

## User Feedback

We can add {py:class}`~autogen_agentchat.agents.UserProxyAgent` to the team to
provide user feedback during a run.
See [Human-in-the-Loop](./tutorial/human-in-the-loop.ipynb) for more details
about {py:class}`~autogen_agentchat.agents.UserProxyAgent`.

To use the {py:class}`~autogen_agentchat.agents.UserProxyAgent` in the 
web search example, we simply add it to the team and update the selector function
to always check for user feedback after the planning agent speaks.
If the user responds with `"APPROVE"`, the conversation continues, otherwise,
the planning agent tries again, until the user approves.
"""
logger.info("## User Feedback")

user_proxy_agent = UserProxyAgent(
    "UserProxyAgent", description="A proxy for the user to approve or disapprove tasks.")


def selector_func_with_user_proxy(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name and messages[-1].source != user_proxy_agent.name:
        return planning_agent.name
    if messages[-1].source == planning_agent.name:
        if messages[-2].source == user_proxy_agent.name and "APPROVE" in messages[-1].content.upper():  # type: ignore
            return None
        return user_proxy_agent.name
    if messages[-1].source == user_proxy_agent.name:
        if "APPROVE" not in messages[-1].content.upper():  # type: ignore
            return planning_agent.name
    return None


async def run_async_code_a5b70700():
    await team.reset()
asyncio.run(run_async_code_a5b70700())
team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent, user_proxy_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    selector_func=selector_func_with_user_proxy,
    allow_repeated_speaker=True,
)


async def run_async_code_ac2575cc():
    await Console(team.run_stream(task=task))
asyncio.run(run_async_code_ac2575cc())

"""
Now, the user's feedback is incorporated into the conversation flow,
and the user can approve or reject the planning agent's decisions.

## Using Reasoning Models

So far in the examples, we have used a `qwen3-1.7b-4bit` model. Models like `qwen3-1.7b-4bit`
and `gemini-1.5-flash` are great at following instructions, so you can
have relatively detailed instructions in the selector prompt for the team and the 
system messages for each agent to guide their behavior.

However, if you are using a reasoning model like `o3-mini`, you will need to
keep the selector prompt and system messages as simple and to the point as possible.
This is because the reasoning models are already good at coming up with their own 
instructions given the context provided to them.

This also means that we don't need a planning agent to break down the task
anymore, since the {py:class}`~autogen_agentchat.teams.SelectorGroupChat` that
uses a reasoning model can do that on its own.

In the following example, we will use `o3-mini` as the model for the
agents and the team, and we will not use a planning agent.
Also, we are keeping the selector prompt and system messages as simple as possible.
"""
logger.info("## Using Reasoning Models")

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""Use web search tool to find information.""",
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""Use tool to perform calculation. If you have not seen the data, ask for it.""",
)

user_proxy_agent = UserProxyAgent(
    "UserProxyAgent",
    description="A user to approve or disapprove tasks.",
)

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
When the task is complete, let the user approve or disapprove the task.
"""

team = SelectorGroupChat(
    [web_search_agent, data_analyst_agent, user_proxy_agent],
    model_client=model_client,
    # Use the same termination condition as before.
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)


async def run_async_code_ac2575cc():
    await Console(team.run_stream(task=task))
asyncio.run(run_async_code_ac2575cc())

"""
```{tip}
For more guidance on how to prompt reasoning models, see the
Azure AI Services Blog on [Prompt Engineering for MLX's O1 and O3-mini Reasoning Models](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/prompt-engineering-for-openai%E2%80%99s-o1-and-o3-mini-reasoning-models/4374010)
```
"""
logger.info("For more guidance on how to prompt reasoning models, see the")

logger.info("\n\n[DONE]", bright=True)
