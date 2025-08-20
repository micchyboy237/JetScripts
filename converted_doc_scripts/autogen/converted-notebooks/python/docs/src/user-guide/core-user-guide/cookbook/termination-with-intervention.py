import asyncio
from autogen_core import (
DefaultInterventionHandler,
DefaultTopicId,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
default_subscription,
message_handler,
)
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import Any
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Termination using Intervention Handler

```{note}
This method is valid when using {py:class}`~autogen_core.SingleThreadedAgentRuntime`.
```

There are many different ways to handle termination in `autogen_core`. Ultimately, the goal is to detect that the runtime no longer needs to be executed and you can proceed to finalization tasks. One way to do this is to use an {py:class}`autogen_core.base.intervention.InterventionHandler` to detect a termination message and then act on it.
"""
logger.info("# Termination using Intervention Handler")



"""
First, we define a dataclass for regular message and message that will be used to signal termination.
"""
logger.info("First, we define a dataclass for regular message and message that will be used to signal termination.")

@dataclass
class Message:
    content: Any


@dataclass
class Termination:
    reason: str

"""
We code our agent to publish a termination message when it decides it is time to terminate.
"""
logger.info("We code our agent to publish a termination message when it decides it is time to terminate.")

@default_subscription
class AnAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")
        self.received = 0

    @message_handler
    async def on_new_message(self, message: Message, ctx: MessageContext) -> None:
        self.received += 1
        if self.received > 3:
            await self.publish_message(Termination(reason="Reached maximum number of messages"), DefaultTopicId())

"""
Next, we create an InterventionHandler that will detect the termination message and act on it. This one hooks into publishes and when it encounters `Termination` it alters its internal state to indicate that termination has been requested.
"""
logger.info("Next, we create an InterventionHandler that will detect the termination message and act on it. This one hooks into publishes and when it encounters `Termination` it alters its internal state to indicate that termination has been requested.")

class TerminationHandler(DefaultInterventionHandler):
    def __init__(self) -> None:
        self._termination_value: Termination | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, Termination):
            self._termination_value = message
        return message

    @property
    def termination_value(self) -> Termination | None:
        return self._termination_value

    @property
    def has_terminated(self) -> bool:
        return self._termination_value is not None

"""
Finally, we add this handler to the runtime and use it to detect termination and stop the runtime when the termination message is received.
"""
logger.info("Finally, we add this handler to the runtime and use it to detect termination and stop the runtime when the termination message is received.")

termination_handler = TerminationHandler()
runtime = SingleThreadedAgentRuntime(intervention_handlers=[termination_handler])

async def run_async_code_b1b3e265():
    await AnAgent.register(runtime, "my_agent", AnAgent)
asyncio.run(run_async_code_b1b3e265())

runtime.start()

async def run_async_code_536bba2d():
    await runtime.publish_message(Message("hello"), DefaultTopicId())
asyncio.run(run_async_code_536bba2d())
async def run_async_code_536bba2d():
    await runtime.publish_message(Message("hello"), DefaultTopicId())
asyncio.run(run_async_code_536bba2d())
async def run_async_code_536bba2d():
    await runtime.publish_message(Message("hello"), DefaultTopicId())
asyncio.run(run_async_code_536bba2d())
async def run_async_code_536bba2d():
    await runtime.publish_message(Message("hello"), DefaultTopicId())
asyncio.run(run_async_code_536bba2d())

async def run_async_code_f9bf1c78():
    await runtime.stop_when(lambda: termination_handler.has_terminated)
asyncio.run(run_async_code_f9bf1c78())

logger.debug(termination_handler.termination_value)

logger.info("\n\n[DONE]", bright=True)