import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
TopicId,
TypeSubscription,
message_handler,
type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Sequential Workflow

Sequential Workflow is a multi-agent design pattern where agents respond in a deterministic sequence. Each agent in the workflow performs a specific task by processing a message, generating a response, and then passing it to the next agent. This pattern is useful for creating deterministic workflows where each agent contributes to a pre-specified sub-task.

In this example, we demonstrate a sequential workflow where multiple agents collaborate to transform a basic product description into a polished marketing copy.

The pipeline consists of four specialized agents:
- **Concept Extractor Agent**: Analyzes the initial product description to extract key features, target audience, and unique selling points (USPs). The output is a structured analysis in a single text block.
- **Writer Agent**: Crafts compelling marketing copy based on the extracted concepts. This agent transforms the analytical insights into engaging promotional content, delivering a cohesive narrative in a single text block.
- **Format & Proof Agent**: Polishes the draft copy by refining grammar, enhancing clarity, and maintaining consistent tone. This agent ensures professional quality and delivers a well-formatted final version.
- **User Agent**: Presents the final, refined marketing copy to the user, completing the workflow.

The following diagram illustrates the sequential workflow in this example:

![Sequential Workflow](sequential-workflow.svg)

We will implement this workflow using publish-subscribe messaging.
Please read about [Topic and Subscription](../core-concepts/topic-and-subscription.md) for the core concepts
and [Broadcast Messaging](../framework/message-and-communication.ipynb#broadcast) for the the API usage.

In this pipeline, agents communicate with each other by publishing their completed work as messages to the topic of the
next agent in the sequence. For example, when the `ConceptExtractor` finishes analyzing the product description, it
publishes its findings to the `"WriterAgent"` topic, which the `WriterAgent` is subscribed to. This pattern continues through
each step of the pipeline, with each agent publishing to the topic that the next agent in line subscribed to.
"""
logger.info("# Sequential Workflow")


"""
## Message Protocol

The message protocol for this example workflow is a simple text message that agents will use to relay their work.
"""
logger.info("## Message Protocol")


@dataclass
class Message:
    content: str


"""
## Topics

Each agent in the workflow will be subscribed to a specific topic type. The topic types are named after the agents in the sequence,
This allows each agent to publish its work to the next agent in the sequence.
"""
logger.info("## Topics")

concept_extractor_topic_type = "ConceptExtractorAgent"
writer_topic_type = "WriterAgent"
format_proof_topic_type = "FormatProofAgent"
user_topic_type = "User"

"""
## Agents

Each agent class is defined with a {py:class}`~autogen_core.type_subscription` decorator to specify the topic type it is subscribed to.
Alternative to the decorator, you can also use the {py:meth}`~autogen_core.AgentRuntime.add_subscription` method to subscribe to a topic through runtime directly.

The concept extractor agent comes up with the initial bullet points for the product description.
"""
logger.info("## Agents")


@type_subscription(topic_type=concept_extractor_topic_type)
class ConceptExtractorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A concept extractor agent.")
        self._system_message = SystemMessage(
            content=(
                "You are a marketing analyst. Given a product description, identify:\n"
                "- Key features\n"
                "- Target audience\n"
                "- Unique selling points\n\n"
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_user_description(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Product description: {message.content}"

        async def async_func_17():
            llm_result = await self._model_client.create(
                messages=[self._system_message, UserMessage(
                    content=prompt, source=self.id.key)],
                cancellation_token=ctx.cancellation_token,
            )
            return llm_result
        llm_result = asyncio.run(async_func_17())
        logger.success(format_json(llm_result))
        response = llm_result.content
        assert isinstance(response, str)
        logger.debug(f"{'-'*80}\n{self.id.type}:\n{response}")

        async def run_async_code_af659285():
            await self.publish_message(Message(response), topic_id=TopicId(writer_topic_type, source=self.id.key))
            return
         = asyncio.run(run_async_code_af659285())
        logger.success(format_json())

"""
The writer agent performs writing.
"""
logger.info("The writer agent performs writing.")

@type_subscription(topic_type=writer_topic_type)
class WriterAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A writer agent.")
        self._system_message = SystemMessage(
            content=(
                "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
                "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
                "Output should be short (around 150 words), output just the copy as a single text block."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_intermediate_text(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Below is the info about the product:\n\n{message.content}"

        async def async_func_17():
            llm_result = await self._model_client.create(
                messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
                cancellation_token=ctx.cancellation_token,
            )
            return llm_result
        llm_result = asyncio.run(async_func_17())
        logger.success(format_json(llm_result))
        response = llm_result.content
        assert isinstance(response, str)
        logger.debug(f"{'-'*80}\n{self.id.type}:\n{response}")

        async def run_async_code_c7b3076e():
            await self.publish_message(Message(response), topic_id=TopicId(format_proof_topic_type, source=self.id.key))
            return 
         = asyncio.run(run_async_code_c7b3076e())
        logger.success(format_json())

"""
The format proof agent performs the formatting.
"""
logger.info("The format proof agent performs the formatting.")

@type_subscription(topic_type=format_proof_topic_type)
class FormatProofAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A format & proof agent.")
        self._system_message = SystemMessage(
            content=(
                "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
                "give format and make it polished. Output the final improved copy as a single text block."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_intermediate_text(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Draft copy:\n{message.content}."
        async def async_func_15():
            llm_result = await self._model_client.create(
                messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
                cancellation_token=ctx.cancellation_token,
            )
            return llm_result
        llm_result = asyncio.run(async_func_15())
        logger.success(format_json(llm_result))
        response = llm_result.content
        assert isinstance(response, str)
        logger.debug(f"{'-'*80}\n{self.id.type}:\n{response}")

        async def run_async_code_98713bb7():
            await self.publish_message(Message(response), topic_id=TopicId(user_topic_type, source=self.id.key))
            return 
         = asyncio.run(run_async_code_98713bb7())
        logger.success(format_json())

"""
In this example, the user agent simply prints the final marketing copy to the console.
In a real-world application, this could be replaced by storing the result to a database, sending an email, or any other desired action.
"""
logger.info("In this example, the user agent simply prints the final marketing copy to the console.")

@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A user agent that outputs the final copy to the user.")

    @message_handler
    async def handle_final_copy(self, message: Message, ctx: MessageContext) -> None:
        logger.debug(f"\n{'-'*80}\n{self.id.type} received final copy:\n{message.content}")

"""
## Workflow

Now we can register the agents to the runtime.
Because we used the {py:class}`~autogen_core.type_subscription` decorator, the runtime will automatically subscribe the agents to the correct topics.
"""
logger.info("## Workflow")

model_client = MLXAutogenChatLLMAdapter(
    model="llama-3.2-3b-instruct",
)

runtime = SingleThreadedAgentRuntime()

await ConceptExtractorAgent.register(
    runtime, type=concept_extractor_topic_type, factory=lambda: ConceptExtractorAgent(model_client=model_client)
)

async def run_async_code_b871bd98():
    await WriterAgent.register(runtime, type=writer_topic_type, factory=lambda: WriterAgent(model_client=model_client))
    return 
 = asyncio.run(run_async_code_b871bd98())
logger.success(format_json())

await FormatProofAgent.register(
    runtime, type=format_proof_topic_type, factory=lambda: FormatProofAgent(model_client=model_client)
)

async def run_async_code_f7b0dbb6():
    await UserAgent.register(runtime, type=user_topic_type, factory=lambda: UserAgent())
    return 
 = asyncio.run(run_async_code_f7b0dbb6())
logger.success(format_json())

"""
## Run the Workflow

Finally, we can run the workflow by publishing a message to the first agent in the sequence.
"""
logger.info("## Run the Workflow")

runtime.start()

await runtime.publish_message(
    Message(content="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours"),
    topic_id=TopicId(concept_extractor_topic_type, source="default"),
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

logger.info("\n\n[DONE]", bright=True)