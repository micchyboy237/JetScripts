import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display  # type: ignore
from autogen_core import (
DefaultTopicId,
FunctionCall,
Image,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
TopicId,
TypeSubscription,
message_handler,
)
from autogen_core.models import (
AssistantMessage,
ChatCompletionClient,
LLMMessage,
SystemMessage,
UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from typing import List
import json
import openai
import os
import string
import uuid

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Group Chat

Group chat is a design pattern where a group of agents share a common thread
of messages: they all subscribe and publish to the same topic. 
Each participant agent is specialized for a particular task, 
such as writer, illustrator, and editor
in a collaborative writing task.
You can also include an agent to represent a human user to help guide the
agents when needed.

In a group chat, participants take turn to publish a message, and the process
is sequential -- only one agent is working at a time.
Under the hood, the order of turns is maintained by a Group Chat Manager agent,
which selects the next agent to speak upon receiving a message.
The exact algorithm for selecting the next agent can vary based on your
application requirements. 
Typically, a round-robin algorithm or a selector with an LLM model is used.

Group chat is useful for dynamically decomposing a complex task into smaller ones 
that can be handled by specialized agents with well-defined roles.
It is also possible to nest group chats into a hierarchy with each participant
a recursive group chat.

In this example, we use AutoGen's Core API to implement the group chat pattern
using event-driven agents.
Please first read about [Topics and Subscriptions](../core-concepts/topic-and-subscription.md)
to understand the concepts and then [Messages and Communication](../framework/message-and-communication.ipynb)
to learn the API usage for pub-sub.
We will demonstrate a simple example of a group chat with a LLM-based selector
for the group chat manager, to create content for a children's story book.

```{note}
While this example illustrates the group chat mechanism, it is complex and
represents a starting point from which you can build your own group chat system
with custom agents and speaker selection algorithms.
The [AgentChat API](../../agentchat-user-guide/index.md) has a built-in implementation
of selector group chat. You can use that if you do not want to use the Core API.
```

We will be using the [rich](https://github.com/Textualize/rich) library to display the messages in a nice format.
"""
logger.info("# Group Chat")





"""
## Message Protocol

The message protocol for the group chat pattern is simple.
1. To start, user or an external agent publishes a `GroupChatMessage` message to the common topic of all participants.
2. The group chat manager selects the next speaker, sends out a `RequestToSpeak` message to that agent.
3. The agent publishes a `GroupChatMessage` message to the common topic upon receiving the `RequestToSpeak` message.
4. This process continues until a termination condition is reached at the group chat manager, which then stops issuing `RequestToSpeak` message, and the group chat ends.

The following diagram illustrates steps 2 to 4 above.

![Group chat message protocol](groupchat.svg)
"""
logger.info("## Message Protocol")

class GroupChatMessage(BaseModel):
    body: UserMessage


class RequestToSpeak(BaseModel):
    pass

"""
## Base Group Chat Agent

Let's first define the agent class that only uses LLM models to generate text.
This is will be used as the base class for all AI agents in the group chat.
"""
logger.info("## Base Group Chat Agent")

class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        Console().logger.debug(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        async def run_async_code_1b41af29():
            async def run_async_code_c1d3ff7b():
                completion = await self._model_client.create([self._system_message] + self._chat_history)
                return completion
            completion = asyncio.run(run_async_code_c1d3ff7b())
            logger.success(format_json(completion))
            return completion
        completion = asyncio.run(run_async_code_1b41af29())
        logger.success(format_json(completion))
        assert isinstance(completion.content, str)
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        Console().logger.debug(Markdown(completion.content))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )

"""
## Writer and Editor Agents

Using the base class, we can define the writer and editor agents with
different system messages.
"""
logger.info("## Writer and Editor Agents")

class WriterAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are a Writer. You produce good work.",
        )


class EditorAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Editor. Plan and guide the task given by the user. Provide critical feedbacks to the draft and illustration produced by Writer and Illustrator. "
            "Approve if the task is completed and the draft and illustration meets user's requirements.",
        )

"""
## Illustrator Agent with Image Generation

Now let's define the `IllustratorAgent` which uses a DALL-E model to generate
an illustration based on the description provided.
We set up the image generator as a tool using {py:class}`~autogen_core.tools.FunctionTool`
wrapper, and use a model client to make the tool call.
"""
logger.info("## Illustrator Agent with Image Generation")

class IllustratorAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        image_client: openai.AsyncClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Illustrator. You use the generate_image tool to create images given user's requirement. "
            "Make sure the images have consistent characters and style.",
        )
        self._image_client = image_client
        self._image_gen_tool = FunctionTool(
            self._image_gen, name="generate_image", description="Call this to generate an image. "
        )

    async def _image_gen(
        self, character_appearence: str, style_attributes: str, worn_and_carried: str, scenario: str
    ) -> str:
        prompt = f"Digital painting of a {character_appearence} character with {style_attributes}. Wearing {worn_and_carried}, {scenario}."
        async def async_func_24():
            response = await self._image_client.images.generate(
                prompt=prompt, model="dall-e-3", response_format="b64_json", size="1024x1024"
            )
            return response
        response = asyncio.run(async_func_24())
        logger.success(format_json(response))
        return response.data[0].b64_json  # type: ignore

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:  # type: ignore
        Console().logger.debug(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        async def async_func_35():
            completion = await self._model_client.create(
                [self._system_message] + self._chat_history,
                tools=[self._image_gen_tool],
                extra_create_args={"tool_choice": "required"},
                cancellation_token=ctx.cancellation_token,
            )
            return completion
        completion = asyncio.run(async_func_35())
        logger.success(format_json(completion))
        assert isinstance(completion.content, list) and all(
            isinstance(item, FunctionCall) for item in completion.content
        )
        images: List[str | Image] = []
        for tool_call in completion.content:
            arguments = json.loads(tool_call.arguments)
            Console().logger.debug(arguments)
            async def run_async_code_9dce6c63():
                async def run_async_code_ff9fba43():
                    result = await self._image_gen_tool.run_json(arguments, ctx.cancellation_token)
                    return result
                result = asyncio.run(run_async_code_ff9fba43())
                logger.success(format_json(result))
                return result
            result = asyncio.run(run_async_code_9dce6c63())
            logger.success(format_json(result))
            image = Image.from_base64(self._image_gen_tool.return_value_as_string(result))
            image = Image.from_pil(image.image.resize((256, 256)))
            display(image.image)  # type: ignore
            images.append(image)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=images, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )

"""
## User Agent

With all the AI agents defined, we can now define the user agent that will
take the role of the human user in the group chat.

The `UserAgent` implementation uses console input to get the user's input.
In a real-world scenario, you can replace this by communicating with a frontend,
and subscribe to responses from the frontend.
"""
logger.info("## User Agent")

class UserAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        pass

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("Enter your message, type 'APPROVE' to conclude the task: ")
        Console().logger.debug(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )

"""
## Group Chat Manager

Lastly, we define the `GroupChatManager` agent which manages the group chat
and selects the next agent to speak using an LLM.
The group chat manager checks if the editor has approved the draft by 
looking for the `"APPORVED"` keyword in the message. If the editor has approved
the draft, the group chat manager stops selecting the next speaker, and the group chat ends.

The group chat manager's constructor takes a list of participants' topic types
as an argument.
To prompt the next speaker to work, 
the `GroupChatManager` agent publishes a `RequestToSpeak` message to the next participant's topic.

In this example, we also make sure the group chat manager always picks a different
participant to speak next, by keeping track of the previous speaker.
This helps to ensure the group chat is not dominated by a single participant.
"""
logger.info("## Group Chat Manager")

class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("approve"):
                return
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        async def run_async_code_ea600906():
            async def run_async_code_ec3f3246():
                completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
                return completion
            completion = asyncio.run(run_async_code_ec3f3246())
            logger.success(format_json(completion))
            return completion
        completion = asyncio.run(run_async_code_ea600906())
        logger.success(format_json(completion))
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                async def run_async_code_5843036a():
                    await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                    return 
                 = asyncio.run(run_async_code_5843036a())
                logger.success(format_json())
                return
        raise ValueError(f"Invalid role selected: {completion.content}")

"""
## Creating the Group Chat

To set up the group chat, we create a {py:class}`~autogen_core.SingleThreadedAgentRuntime`
and register the agents' factories and subscriptions.

Each participant agent subscribes to both the group chat topic as well as its own
topic in order to receive `RequestToSpeak` messages, 
while the group chat manager agent only subcribes to the group chat topic.
"""
logger.info("## Creating the Group Chat")

runtime = SingleThreadedAgentRuntime()

editor_topic_type = "Editor"
writer_topic_type = "Writer"
illustrator_topic_type = "Illustrator"
user_topic_type = "User"
group_chat_topic_type = "group_chat"

editor_description = "Editor for planning and reviewing the content."
writer_description = "Writer for creating any text content."
user_description = "User for providing final approval."
illustrator_description = "An illustrator for creating images."

model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
)

async def async_func_17():
    editor_agent_type = await EditorAgent.register(
        runtime,
        editor_topic_type,  # Using topic type as the agent type.
        lambda: EditorAgent(
            description=editor_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
        ),
    )
    return editor_agent_type
editor_agent_type = asyncio.run(async_func_17())
logger.success(format_json(editor_agent_type))
async def run_async_code_6a1b9d19():
    await runtime.add_subscription(TypeSubscription(topic_type=editor_topic_type, agent_type=editor_agent_type.type))
    return 
 = asyncio.run(run_async_code_6a1b9d19())
logger.success(format_json())
async def run_async_code_c3cc96e8():
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=editor_agent_type.type))
    return 
 = asyncio.run(run_async_code_c3cc96e8())
logger.success(format_json())

async def async_func_29():
    writer_agent_type = await WriterAgent.register(
        runtime,
        writer_topic_type,  # Using topic type as the agent type.
        lambda: WriterAgent(
            description=writer_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
        ),
    )
    return writer_agent_type
writer_agent_type = asyncio.run(async_func_29())
logger.success(format_json(writer_agent_type))
async def run_async_code_9f747b37():
    await runtime.add_subscription(TypeSubscription(topic_type=writer_topic_type, agent_type=writer_agent_type.type))
    return 
 = asyncio.run(run_async_code_9f747b37())
logger.success(format_json())
async def run_async_code_51e4e8e9():
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=writer_agent_type.type))
    return 
 = asyncio.run(run_async_code_51e4e8e9())
logger.success(format_json())

async def async_func_41():
    illustrator_agent_type = await IllustratorAgent.register(
        runtime,
        illustrator_topic_type,
        lambda: IllustratorAgent(
            description=illustrator_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            image_client=openai.AsyncClient(
            ),
        ),
    )
    return illustrator_agent_type
illustrator_agent_type = asyncio.run(async_func_41())
logger.success(format_json(illustrator_agent_type))
await runtime.add_subscription(
    TypeSubscription(topic_type=illustrator_topic_type, agent_type=illustrator_agent_type.type)
)
await runtime.add_subscription(
    TypeSubscription(topic_type=group_chat_topic_type, agent_type=illustrator_agent_type.type)
)

async def async_func_59():
    user_agent_type = await UserAgent.register(
        runtime,
        user_topic_type,
        lambda: UserAgent(description=user_description, group_chat_topic_type=group_chat_topic_type),
    )
    return user_agent_type
user_agent_type = asyncio.run(async_func_59())
logger.success(format_json(user_agent_type))
async def run_async_code_3a3d930f():
    await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))
    return 
 = asyncio.run(run_async_code_3a3d930f())
logger.success(format_json())
async def run_async_code_5306bf3c():
    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=user_agent_type.type))
    return 
 = asyncio.run(run_async_code_5306bf3c())
logger.success(format_json())

async def async_func_67():
    group_chat_manager_type = await GroupChatManager.register(
        runtime,
        "group_chat_manager",
        lambda: GroupChatManager(
            participant_topic_types=[writer_topic_type, illustrator_topic_type, editor_topic_type, user_topic_type],
            model_client=model_client,
            participant_descriptions=[writer_description, illustrator_description, editor_description, user_description],
        ),
    )
    return group_chat_manager_type
group_chat_manager_type = asyncio.run(async_func_67())
logger.success(format_json(group_chat_manager_type))
await runtime.add_subscription(
    TypeSubscription(topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type)
)

"""
## Running the Group Chat

We start the runtime and publish a `GroupChatMessage` for the task to start the group chat.
"""
logger.info("## Running the Group Chat")

runtime.start()
session_id = str(uuid.uuid4())
await runtime.publish_message(
    GroupChatMessage(
        body=UserMessage(
            content="Please write a short story about the gingerbread man with up to 3 photo-realistic illustrations.",
            source="User",
        )
    ),
    TopicId(type=group_chat_topic_type, source=session_id),
)
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
    return 
 = asyncio.run(run_async_code_b7ca34d4())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
From the output, you can see the writer, illustrator, and editor agents
taking turns to speak and collaborate to generate a picture book, before
asking for final approval from the user.

## Next Steps

This example showcases a simple implementation of the group chat pattern -- 
**it is not meant to be used in real applications.** You can improve the
speaker selection algorithm. For example, you can avoid using LLM when simple
rules are sufficient and more reliable: 
you can use a rule that the editor always speaks after the writer.

The [AgentChat API](../../agentchat-user-guide/index.md) provides a high-level
API for selector group chat. It has more features but mostly shares the same
design as this implementation.
"""
logger.info("## Next Steps")

logger.info("\n\n[DONE]", bright=True)