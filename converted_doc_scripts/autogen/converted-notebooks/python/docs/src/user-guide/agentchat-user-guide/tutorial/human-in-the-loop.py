import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Human-in-the-Loop

In the previous section [Teams](./teams.ipynb), we have seen how to create, observe,
and control a team of agents.
This section will focus on how to interact with the team from your application,
and provide human feedback to the team.

There are two main ways to interact with the team from your application:

1. During a team's run -- execution of {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` or {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream`, provide feedback through a {py:class}`~autogen_agentchat.agents.UserProxyAgent`.
2. Once the run terminates, provide feedback through input to the next call to {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` or {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run_stream`.

We will cover both methods in this section.

To jump straight to code samples on integration with web and UI frameworks, see the following links:
- [AgentChat + FastAPI](https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_fastapi)
- [AgentChat + ChainLit](https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_chainlit)
- [AgentChat + Streamlit](https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_streamlit)

## Providing Feedback During a Run

The {py:class}`~autogen_agentchat.agents.UserProxyAgent` is a special built-in agent
that acts as a proxy for a user to provide feedback to the team.

To use the {py:class}`~autogen_agentchat.agents.UserProxyAgent`, you can create an instance of it
and include it in the team before running the team.
The team will decide when to call the {py:class}`~autogen_agentchat.agents.UserProxyAgent`
to ask for feedback from the user.

For example in a {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` team, 
the {py:class}`~autogen_agentchat.agents.UserProxyAgent` is called in the order
in which it is passed to the team, while in a {py:class}`~autogen_agentchat.teams.SelectorGroupChat`
team, the selector prompt or selector function determines when the 
{py:class}`~autogen_agentchat.agents.UserProxyAgent` is called.

The following diagram illustrates how you can use 
{py:class}`~autogen_agentchat.agents.UserProxyAgent`
to get feedback from the user during a team's run:

![human-in-the-loop-user-proxy](./human-in-the-loop-user-proxy.svg)

The bold arrows indicates the flow of control during a team's run:
when the team calls the {py:class}`~autogen_agentchat.agents.UserProxyAgent`,
it transfers the control to the application/user, and waits for the feedback;
once the feedback is provided, the control is transferred back to the team
and the team continues its execution.

```{note}
When {py:class}`~autogen_agentchat.agents.UserProxyAgent` is called during a run,
it blocks the execution of the team until the user provides feedback or errors out.
This will hold up the team's progress and put the team in an unstable state
that cannot be saved or resumed.
```

Due to the blocking nature of this approach, it is recommended to use it only for short interactions
that require immediate feedback from the user, such as asking for approval or disapproval
with a button click, or an alert requiring immediate attention otherwise failing the task.

Here is an example of how to use the {py:class}`~autogen_agentchat.agents.UserProxyAgent`
in a {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` for a poetry generation task:
"""
logger.info("# Human-in-the-Loop")


model_client = OllamaChatCompletionClient(model="llama3.1")
assistant = AssistantAgent("assistant", model_client=model_client)
user_proxy = UserProxyAgent("user_proxy", input_func=input)  # Use input() to get user input from console.

termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)

stream = team.run_stream(task="Write a 4-line poem about the ocean.")
async def run_async_code_71db6073():
    await Console(stream)
    return 
 = asyncio.run(run_async_code_71db6073())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
From the console output, you can see the team solicited feedback from the user
through `user_proxy` to approve the generated poem.

You can provide your own input function to the {py:class}`~autogen_agentchat.agents.UserProxyAgent`
to customize the feedback process.
For example, when the team is running as a web service, you can use a custom
input function to wait for message from a web socket connection.
The following code snippet shows an example of custom input function
when using the [FastAPI](https://fastapi.tiangolo.com/) web framework:

```python
@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
        data = await websocket.receive_json() # Wait for user message from websocket.
        message = TextMessage.model_validate(data) # Assume user message is a TextMessage.
        return message.content
    
    # Create user proxy with custom input function
    # Run the team with the user proxy
    # ...
```

See the [AgentChat FastAPI sample](https://github.com/microsoft/autogen/blob/main/python/samples/agentchat_fastapi) for a complete example.

For [ChainLit](https://github.com/Chainlit/chainlit) integration with {py:class}`~autogen_agentchat.agents.UserProxyAgent`,
see the [AgentChat ChainLit sample](https://github.com/microsoft/autogen/blob/main/python/samples/agentchat_chainlit).

## Providing Feedback to the Next Run

Often times, an application or a user interacts with the team of agents in an interactive loop:
the team runs until termination, 
the application or user provides feedback, and the team runs again with the feedback.

This approach is useful in a persisted session
with asynchronous communication between the team and the application/user:
Once a team finishes a run, the application saves the state of the team,
puts it in a persistent storage, and resumes the team when the feedback arrives.

```{note}
For how to save and load the state of a team, please refer to [Managing State](./state.ipynb).
This section will focus on the feedback mechanisms.
```

The following diagram illustrates the flow of control in this approach:

![human-in-the-loop-termination](./human-in-the-loop-termination.svg)

There are two ways to implement this approach:

- Set the maximum number of turns so that the team always stops after the specified number of turns.
- Use termination conditions such as {py:class}`~autogen_agentchat.conditions.TextMentionTermination` and {py:class}`~autogen_agentchat.conditions.HandoffTermination` to allow the team to decide when to stop and give control back, given the team's internal state.

You can use both methods together to achieve your desired behavior.

### Using Max Turns

This method allows you to pause the team for user input by setting a maximum number of turns. For instance, you can configure the team to stop after the first agent responds by setting `max_turns` to 1. This is particularly useful in scenarios where continuous user engagement is required, such as in a chatbot.

To implement this, set the `max_turns` parameter in the {py:meth}`~autogen_agentchat.teams.RoundRobinGroupChat` constructor.

```python
team = RoundRobinGroupChat([...], max_turns=1)
```

Once the team stops, the turn count will be reset. When you resume the team,
it will start from 0 again. However, the team's internal state will be preserved,
for example, the {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` will
resume from the next agent in the list with the same conversation history.

```{note}
`max_turn` is specific to the team class and is currently only supported by
{py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`, {py:class}`~autogen_agentchat.teams.SelectorGroupChat`, and {py:class}`~autogen_agentchat.teams.Swarm`.
When used with termination conditions, the team will stop when either condition is met.
```

Here is an example of how to use `max_turns` in a {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` for a poetry generation task
with a maximum of 1 turn:
"""
logger.info("# Create user proxy with custom input function")


model_client = OllamaChatCompletionClient(model="llama3.1")
assistant = AssistantAgent("assistant", model_client=model_client)

team = RoundRobinGroupChat([assistant], max_turns=1)

task = "Write a 4-line poem about the ocean."
while True:
    stream = team.run_stream(task=task)
    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return 
     = asyncio.run(run_async_code_8cdf6b5b())
    logger.success(format_json())
    task = input("Enter your feedback (type 'exit' to leave): ")
    if task.lower().strip() == "exit":
        break
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
You can see that the team stopped immediately after one agent responded.

### Using Termination Conditions

We have already seen several examples of termination conditions in the previous sections.
In this section, we focus on {py:class}`~autogen_agentchat.conditions.HandoffTermination`
which stops the team when an agent sends a {py:class}`~autogen_agentchat.messages.HandoffMessage` message.

Let's create a team with a single {py:class}`~autogen_agentchat.agents.AssistantAgent` agent
with a handoff setting, and run the team with a task that requires additional input from the user
because the agent doesn't have relevant tools to continue processing the task.

```{note}
The model used with {py:class}`~autogen_agentchat.agents.AssistantAgent` must support tool call
to use the handoff feature.
```
"""
logger.info("### Using Termination Conditions")


model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
)

lazy_agent = AssistantAgent(
    "lazy_assistant",
    model_client=model_client,
    handoffs=[Handoff(target="user", message="Transfer to user.")],
    system_message="If you cannot complete the task, transfer to user. Otherwise, when finished, respond with 'TERMINATE'.",
)

handoff_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")

lazy_agent_team = RoundRobinGroupChat([lazy_agent], termination_condition=handoff_termination | text_termination)

task = "What is the weather in New York?"
async def run_async_code_3b0e12b4():
    await Console(lazy_agent_team.run_stream(task=task), output_stats=True)
    return 
 = asyncio.run(run_async_code_3b0e12b4())
logger.success(format_json())

"""
You can see the team stopped due to the handoff message was detected.
Let's continue the team by providing the information the agent needs.
"""
logger.info("You can see the team stopped due to the handoff message was detected.")

async def run_async_code_69ec9576():
    await Console(lazy_agent_team.run_stream(task="The weather in New York is sunny."))
    return 
 = asyncio.run(run_async_code_69ec9576())
logger.success(format_json())

"""
You can see the team continued after the user provided the information.

```{note}
If you are using {py:class}`~autogen_agentchat.teams.Swarm` team with
{py:class}`~autogen_agentchat.conditions.HandoffTermination` targeting user,
to resume the team, you need to set the `task` to a {py:class}`~autogen_agentchat.messages.HandoffMessage`
with the `target` set to the next agent you want to run.
See [Swarm](../swarm.ipynb) for more details.
```
"""
logger.info("You can see the team continued after the user provided the information.")

logger.info("\n\n[DONE]", bright=True)