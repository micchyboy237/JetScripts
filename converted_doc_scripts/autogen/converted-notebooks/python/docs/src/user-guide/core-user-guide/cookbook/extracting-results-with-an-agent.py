import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
ClosureAgent,
ClosureContext,
DefaultSubscription,
DefaultTopicId,
MessageContext,
SingleThreadedAgentRuntime,
)
from dataclasses import dataclass
from jet.logger import CustomLogger
import asyncio
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Extracting Results with an Agent

When running a multi-agent system to solve some task, you may want to extract the result of the system once it has reached termination. This guide showcases one way to achieve this. Given that agent instances are not directly accessible from the outside, we will use an agent to publish the final result to an accessible location.

If you model your system to publish some `FinalResult` type then you can create an agent whose sole job is to subscribe to this and make it available externally. For simple agents like this the {py:class}`~autogen_core.components.ClosureAgent` is an option to reduce the amount of boilerplate code. This allows you to define a function that will be associated as the agent's message handler. In this example, we're going to use a queue shared between the agent and the external code to pass the result.

```{note}
When considering how to extract results from a multi-agent system, you must always consider the subscriptions of the agent and the topics they publish to.
This is because the agent will only receive messages from topics it is subscribed to.
```
"""
logger.info("# Extracting Results with an Agent")



"""
Define a dataclass for the final result.
"""
logger.info("Define a dataclass for the final result.")

@dataclass
class FinalResult:
    value: str

"""
Create a queue to pass the result from the agent to the external code.
"""
logger.info("Create a queue to pass the result from the agent to the external code.")

queue = asyncio.Queue[FinalResult]()

"""
Create a function closure for outputting the final result to the queue.
The function must follow the signature
`Callable[[AgentRuntime, AgentId, T, MessageContext], Awaitable[Any]]`
where `T` is the type of the message the agent will receive.
You can use union types to handle multiple message types.
"""
logger.info("Create a function closure for outputting the final result to the queue.")

async def output_result(_agent: ClosureContext, message: FinalResult, ctx: MessageContext) -> None:
    async def run_async_code_b3b51e6f():
        await queue.put(message)
        return 
     = asyncio.run(run_async_code_b3b51e6f())
    logger.success(format_json())

"""
Let's create a runtime and register a {py:class}`~autogen_core.components.ClosureAgent` that will publish the final result to the queue.
"""
logger.info("Let's create a runtime and register a {py:class}`~autogen_core.components.ClosureAgent` that will publish the final result to the queue.")

runtime = SingleThreadedAgentRuntime()
await ClosureAgent.register_closure(
    runtime, "output_result", output_result, subscriptions=lambda: [DefaultSubscription()]
)

"""
We can simulate the collection of final results by publishing them directly to the runtime.
"""
logger.info("We can simulate the collection of final results by publishing them directly to the runtime.")

runtime.start()
async def run_async_code_6e97825e():
    await runtime.publish_message(FinalResult("Result 1"), DefaultTopicId())
    return 
 = asyncio.run(run_async_code_6e97825e())
logger.success(format_json())
async def run_async_code_baba95bc():
    await runtime.publish_message(FinalResult("Result 2"), DefaultTopicId())
    return 
 = asyncio.run(run_async_code_baba95bc())
logger.success(format_json())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
    return 
 = asyncio.run(run_async_code_b7ca34d4())
logger.success(format_json())

"""
We can take a look at the queue to see the final result.
"""
logger.info("We can take a look at the queue to see the final result.")

while not queue.empty():
    async def run_async_code_e348b90b():
        async def run_async_code_6fecdaa7():
            logger.debug((result := await queue.get()).value)
            return logger.debug((result :
        logger.debug((result : = asyncio.run(run_async_code_6fecdaa7())
        logger.success(format_json(logger.debug((result :))
        return logger.debug((result :
    logger.debug((result : = asyncio.run(run_async_code_e348b90b())
    logger.success(format_json(logger.debug((result :))

logger.info("\n\n[DONE]", bright=True)