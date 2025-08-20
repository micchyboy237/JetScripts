import asyncio
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import Callable
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Quick Start

:::{note}
See [here](installation) for installation instructions.
:::

Before diving into the core APIs, let's start with a simple example of two agents that count down from 10 to 1.

We first define the agent classes and their respective procedures for 
handling messages.
We create two agent classes: `Modifier` and `Checker`. The `Modifier` agent modifies a number that is given and the `Check` agent checks the value against a condition.
We also create a `Message` data class, which defines the messages that are passed between the agents.
"""
logger.info("# Quick Start")




@dataclass
class Message:
    content: int


@default_subscription
class Modifier(RoutedAgent):
    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self._modify_val = modify_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        val = self._modify_val(message.content)
        logger.debug(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())  # type: ignore


@default_subscription
class Checker(RoutedAgent):
    def __init__(self, run_until: Callable[[int], bool]) -> None:
        super().__init__("A checker agent.")
        self._run_until = run_until

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if not self._run_until(message.content):
            logger.debug(f"{'-'*80}\nChecker:\n{message.content} passed the check, continue.")
            await self.publish_message(Message(content=message.content), DefaultTopicId())
        else:
            logger.debug(f"{'-'*80}\nChecker:\n{message.content} failed the check, stopping.")

"""
You might have already noticed, the agents' logic, whether it is using model or code executor,
is completely decoupled from
how messages are delivered. This is the core idea: the framework provides
a communication infrastructure, and the agents are responsible for their own
logic. We call the communication infrastructure an **Agent Runtime**.

Agent runtime is a key concept of this framework. Besides delivering messages,
it also manages agents' lifecycle. 
So the creation of agents are handled by the runtime.

The following code shows how to register and run the agents using 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
a local embedded agent runtime implementation.

```{note}
If you are using VSCode or other Editor remember to import asyncio and wrap the code with async def main() -> None: and run the code with asyncio.run(main()) function.
```
"""
logger.info("You might have already noticed, the agents' logic, whether it is using model or code executor,")


runtime = SingleThreadedAgentRuntime()

async def async_func_4():
    await Modifier.register(
        runtime,
        "modifier",
        lambda: Modifier(modify_val=lambda x: x - 1),
    )
asyncio.run(async_func_4())

async def async_func_10():
    await Checker.register(
        runtime,
        "checker",
        lambda: Checker(run_until=lambda x: x <= 1),
    )
asyncio.run(async_func_10())

runtime.start()
async def run_async_code_e1e147d5():
    await runtime.send_message(Message(10), AgentId("checker", "default"))
asyncio.run(run_async_code_e1e147d5())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
From the agent's output, we can see the value was successfully decremented from 10 to 1 as the modifier and checker conditions dictate.

AutoGen also supports a distributed agent runtime, which can host agents running on
different processes or machines, with different identities, languages and dependencies.

To learn how to use agent runtime, communication, message handling, and subscription, please continue
reading the sections following this quick start.
"""
logger.info("From the agent's output, we can see the value was successfully decremented from 10 to 1 as the modifier and checker conditions dictate.")

logger.info("\n\n[DONE]", bright=True)