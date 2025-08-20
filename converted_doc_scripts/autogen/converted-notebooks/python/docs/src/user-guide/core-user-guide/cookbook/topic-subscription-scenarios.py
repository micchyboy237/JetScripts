import asyncio
from autogen_core import (
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
TopicId,
TypeSubscription,
message_handler,
)
from autogen_core._default_subscription import DefaultSubscription
from autogen_core._default_topic import DefaultTopicId
from autogen_core.models import (
SystemMessage,
)
from dataclasses import dataclass
from enum import Enum
from jet.logger import CustomLogger
from typing import List
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
## Topic and Subscription Example Scenarios

### Introduction

In this cookbook, we explore how broadcasting works for agent communication in AutoGen using four different broadcasting scenarios. These scenarios illustrate various ways to handle and distribute messages among agents. We'll use a consistent example of a tax management company processing client requests to demonstrate each scenario.

### Scenario Overview

Imagine a tax management company that offers various services to clients, such as tax planning, dispute resolution, compliance, and preparation. The company employs a team of tax specialists, each with expertise in one of these areas, and a tax system manager who oversees the operations.

Clients submit requests that need to be processed by the appropriate specialists. The communication between the clients, the tax system manager, and the tax specialists is handled through broadcasting in this system.

We'll explore how different broadcasting scenarios affect the way messages are distributed among agents and how they can be used to tailor the communication flow to specific needs.

---

### Broadcasting Scenarios Overview

We will cover the following broadcasting scenarios:

1. **Single-Tenant, Single Scope of Publishing**
2. **Multi-Tenant, Single Scope of Publishing**
3. **Single-Tenant, Multiple Scopes of Publishing**
4. **Multi-Tenant, Multiple Scopes of Publishing**


Each scenario represents a different approach to message distribution and agent interaction within the system. By understanding these scenarios, you can design agent communication strategies that best fit your application's requirements.
"""
logger.info("## Topic and Subscription Example Scenarios")



class TaxSpecialty(str, Enum):
    PLANNING = "planning"
    DISPUTE_RESOLUTION = "dispute_resolution"
    COMPLIANCE = "compliance"
    PREPARATION = "preparation"


@dataclass
class ClientRequest:
    content: str


@dataclass
class RequestAssessment:
    content: str


class TaxSpecialist(RoutedAgent):
    def __init__(
        self,
        description: str,
        specialty: TaxSpecialty,
        system_messages: List[SystemMessage],
    ) -> None:
        super().__init__(description)
        self.specialty = specialty
        self._system_messages = system_messages
        self._memory: List[ClientRequest] = []

    @message_handler
    async def handle_message(self, message: ClientRequest, ctx: MessageContext) -> None:
        logger.debug(f"\n{'='*50}\nTax specialist {self.id} with specialty {self.specialty}:\n{message.content}")
        if ctx.topic_id is None:
            raise ValueError("Topic ID is required for broadcasting")
        await self.publish_message(
            message=RequestAssessment(content=f"I can handle this request in {self.specialty}."),
            topic_id=ctx.topic_id,
        )

"""
### 1. Single-Tenant, Single Scope of Publishing

#### Scenarios Explanation
In the single-tenant, single scope of publishing scenario:

- All agents operate within a single tenant (e.g., one client or user session).
- Messages are published to a single topic, and all agents subscribe to this topic.
- Every agent receives every message that gets published to the topic.

This scenario is suitable for situations where all agents need to be aware of all messages, and there's no need to isolate communication between different groups of agents or sessions.

#### Application in the Tax Specialist Company

In our tax specialist company, this scenario implies:

- All tax specialists receive every client request and internal message.
- All agents collaborate closely, with full visibility of all communications.
- Useful for tasks or teams where all agents need to be aware of all messages.

#### How the Scenario Works

- Subscriptions: All agents use the default subscription(e.g., "default").
- Publishing: Messages are published to the default topic.
- Message Handling: Each agent decides whether to act on a message based on its content and available handlers.

#### Benefits
- Simplicity: Easy to set up and understand.
- Collaboration: Promotes transparency and collaboration among agents.
- Flexibility: Agents can dynamically decide which messages to process.

#### Considerations
- Scalability: May not scale well with a large number of agents or messages.
- Efficiency: Agents may receive many irrelevant messages, leading to unnecessary processing.
"""
logger.info("### 1. Single-Tenant, Single Scope of Publishing")

async def run_single_tenant_single_scope() -> None:
    runtime = SingleThreadedAgentRuntime()

    specialist_agent_type_1 = "TaxSpecialist_1"
    specialist_agent_type_2 = "TaxSpecialist_2"
    async def async_func_5():
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type_1,
            factory=lambda: TaxSpecialist(
                description="A tax specialist 1",
                specialty=TaxSpecialty.PLANNING,
                system_messages=[SystemMessage(content="You are a tax specialist.")],
            ),
        )
    asyncio.run(async_func_5())

    async def async_func_15():
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type_2,
            factory=lambda: TaxSpecialist(
                description="A tax specialist 2",
                specialty=TaxSpecialty.DISPUTE_RESOLUTION,
                system_messages=[SystemMessage(content="You are a tax specialist.")],
            ),
        )
    asyncio.run(async_func_15())

    async def run_async_code_26809051():
        await runtime.add_subscription(DefaultSubscription(agent_type=specialist_agent_type_1))
    asyncio.run(run_async_code_26809051())
    async def run_async_code_56d331b8():
        await runtime.add_subscription(DefaultSubscription(agent_type=specialist_agent_type_2))
    asyncio.run(run_async_code_56d331b8())

    runtime.start()
    async def run_async_code_51965411():
        await runtime.publish_message(ClientRequest("I need to have my tax for 2024 prepared."), topic_id=DefaultTopicId())
    asyncio.run(run_async_code_51965411())
    async def run_async_code_3b466eef():
        await runtime.stop_when_idle()
    asyncio.run(run_async_code_3b466eef())


async def run_async_code_b44d9695():
    await run_single_tenant_single_scope()
asyncio.run(run_async_code_b44d9695())

"""
### 2. Multi-Tenant, Single Scope of Publishing

#### Scenario Explanation

In the multi-tenant, single scope of publishing scenario:

- There are multiple tenants (e.g., multiple clients or user sessions).
- Each tenant has its own isolated topic through the topic source.
- All agents within a tenant subscribe to the tenant's topic. If needed, new agent instances are created for each tenant.
- Messages are only visible to agents within the same tenant.

This scenario is useful when you need to isolate communication between different tenants but want all agents within a tenant to be aware of all messages.

#### Application in the Tax Specialist Company

In this scenario:

- The company serves multiple clients (tenants) simultaneously.
- For each client, a dedicated set of agent instances is created.
- Each client's communication is isolated from others.
- All agents for a client receive messages published to that client's topic.

#### How the Scenario Works

- Subscriptions: Agents subscribe to topics based on the tenant's identity.
- Publishing: Messages are published to the tenant-specific topic.
- Message Handling: Agents only receive messages relevant to their tenant.

#### Benefits
- Tenant Isolation: Ensures data privacy and separation between clients.
- Collaboration Within Tenant: Agents can collaborate freely within their tenant.

#### Considerations
- Complexity: Requires managing multiple sets of agents and topics.
- Resource Usage: More agent instances may consume additional resources.
"""
logger.info("### 2. Multi-Tenant, Single Scope of Publishing")

async def run_multi_tenant_single_scope() -> None:
    runtime = SingleThreadedAgentRuntime()

    tenants = ["ClientABC", "ClientXYZ"]

    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        async def async_func_7():
            await TaxSpecialist.register(
                runtime=runtime,
                type=specialist_agent_type,
                factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                    description=f"A tax specialist in {specialty.value}.",
                    specialty=specialty,
                    system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
                ),
            )
        asyncio.run(async_func_7())
        specialist_subscription = DefaultSubscription(agent_type=specialist_agent_type)
        async def run_async_code_d34a487f():
            await runtime.add_subscription(specialist_subscription)
        asyncio.run(run_async_code_d34a487f())

    runtime.start()

    for tenant in tenants:
        topic_source = tenant  # The topic source is the client name
        topic_id = DefaultTopicId(source=topic_source)
        async def async_func_24():
            await runtime.publish_message(
                ClientRequest(f"{tenant} requires tax services."),
                topic_id=topic_id,
            )
        asyncio.run(async_func_24())

    async def run_async_code_acb6afe5():
        await asyncio.sleep(1)
    asyncio.run(run_async_code_acb6afe5())

    async def run_async_code_3b466eef():
        await runtime.stop_when_idle()
    asyncio.run(run_async_code_3b466eef())


async def run_async_code_72a9c1c9():
    await run_multi_tenant_single_scope()
asyncio.run(run_async_code_72a9c1c9())

"""
### 3. Single-Tenant, Multiple Scopes of Publishing

#### Scenario Explanation

In the single-tenant, multiple scopes of publishing scenario:

- All agents operate within a single tenant.
- Messages are published to different topics.
- Agents subscribe to specific topics relevant to their role or specialty.
- Messages are directed to subsets of agents based on the topic.

This scenario allows for targeted communication within a tenant, enabling more granular control over message distribution.

#### Application in the Tax Management Company

In this scenario:

- The tax system manager communicates with specific specialists based on their specialties.
- Different topics represent different specialties (e.g., "planning", "compliance").
- Specialists subscribe only to the topic that matches their specialty.
- The manager publishes messages to specific topics to reach the intended specialists.

#### How the Scenario Works

- Subscriptions: Agents subscribe to topics corresponding to their specialties.
- Publishing: Messages are published to topics based on the intended recipients.
- Message Handling: Only agents subscribed to a topic receive its messages.
#### Benefits

- Targeted Communication: Messages reach only the relevant agents.
- Efficiency: Reduces unnecessary message processing by agents.

#### Considerations

- Setup Complexity: Requires careful management of topics and subscriptions.
- Flexibility: Changes in communication scenarios may require updating subscriptions.
"""
logger.info("### 3. Single-Tenant, Multiple Scopes of Publishing")

async def run_single_tenant_multiple_scope() -> None:
    runtime = SingleThreadedAgentRuntime()
    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type,
            factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                description=f"A tax specialist in {specialty.value}.",
                specialty=specialty,
                system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
            ),
        )
        specialist_subscription = TypeSubscription(topic_type=specialty.value, agent_type=specialist_agent_type)
        await runtime.add_subscription(specialist_subscription)

    runtime.start()

    for specialty in TaxSpecialty:
        topic_id = TopicId(type=specialty.value, source="default")
        async def async_func_20():
            await runtime.publish_message(
                ClientRequest(f"I need assistance with {specialty.value} taxes."),
                topic_id=topic_id,
            )
        asyncio.run(async_func_20())

    async def run_async_code_acb6afe5():
        await asyncio.sleep(1)
    asyncio.run(run_async_code_acb6afe5())

    async def run_async_code_3b466eef():
        await runtime.stop_when_idle()
    asyncio.run(run_async_code_3b466eef())


async def run_async_code_86fc9be7():
    await run_single_tenant_multiple_scope()
asyncio.run(run_async_code_86fc9be7())

"""
### 4. Multi-Tenant, Multiple Scopes of Publishing

#### Scenario Explanation

In the multi-tenant, multiple scopes of publishing scenario:

- There are multiple tenants, each with their own set of agents.
- Messages are published to multiple topics within each tenant.
- Agents subscribe to tenant-specific topics relevant to their role.
- Combines tenant isolation with targeted communication.

This scenario provides the highest level of control over message distribution, suitable for complex systems with multiple clients and specialized communication needs.

#### Application in the Tax Management Company

In this scenario:

- The company serves multiple clients, each with dedicated agent instances.
- Within each client, agents communicate using multiple topics based on specialties.
- For example, Client A's planning specialist subscribes to the "planning" topic with source "ClientA".
- The tax system manager for each client communicates with their specialists using tenant-specific topics.

#### How the Scenario Works

- Subscriptions: Agents subscribe to topics based on both tenant identity and specialty.
- Publishing: Messages are published to tenant-specific and specialty-specific topics.
- Message Handling: Only agents matching the tenant and topic receive messages.

#### Benefits

- Complete Isolation: Ensures both tenant and communication isolation.
- Granular Control: Enables precise routing of messages to intended agents.

#### Considerations

- Complexity: Requires careful management of topics, tenants, and subscriptions.
- Resource Usage: Increased number of agent instances and topics may impact resources.
"""
logger.info("### 4. Multi-Tenant, Multiple Scopes of Publishing")

async def run_multi_tenant_multiple_scope() -> None:
    runtime = SingleThreadedAgentRuntime()

    tenants = ["ClientABC", "ClientXYZ"]

    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        async def async_func_7():
            await TaxSpecialist.register(
                runtime=runtime,
                type=specialist_agent_type,
                factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                    description=f"A tax specialist in {specialty.value}.",
                    specialty=specialty,
                    system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
                ),
            )
        asyncio.run(async_func_7())
        for tenant in tenants:
            specialist_subscription = TypeSubscription(
                topic_type=f"{tenant}_{specialty.value}", agent_type=specialist_agent_type
            )
            async def run_async_code_353886c4():
                await runtime.add_subscription(specialist_subscription)
            asyncio.run(run_async_code_353886c4())

    runtime.start()

    for tenant in tenants:
        for specialty in TaxSpecialty:
            topic_id = TopicId(type=f"{tenant}_{specialty.value}", source=tenant)
            async def async_func_27():
                await runtime.publish_message(
                    ClientRequest(f"{tenant} needs assistance with {specialty.value} taxes."),
                    topic_id=topic_id,
                )
            asyncio.run(async_func_27())

    async def run_async_code_acb6afe5():
        await asyncio.sleep(1)
    asyncio.run(run_async_code_acb6afe5())

    async def run_async_code_3b466eef():
        await runtime.stop_when_idle()
    asyncio.run(run_async_code_3b466eef())


async def run_async_code_39f3bcbf():
    await run_multi_tenant_multiple_scope()
asyncio.run(run_async_code_39f3bcbf())

logger.info("\n\n[DONE]", bright=True)