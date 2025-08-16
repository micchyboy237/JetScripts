import asyncio
from jet.transformers.formatters import format_json
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost
from dataclasses import dataclass
from jet.logger import CustomLogger
import asyncio
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Distributed Agent Runtime

```{attention}
The distributed agent runtime is an experimental feature. Expect breaking changes
to the API.
```

A distributed agent runtime facilitates communication and agent lifecycle management
across process boundaries.
It consists of a host service and at least one worker runtime.

The host service maintains connections to all active worker runtimes,
facilitates message delivery, and keeps sessions for all direct messages (i.e., RPCs).
A worker runtime processes application code (agents) and connects to the host service.
It also advertises the agents which they support to the host service,
so the host service can deliver messages to the correct worker.

````{note}
The distributed agent runtime requires extra dependencies, install them using:
```bash
pip install "autogen-ext[grpc]"
```
````

We can start a host service using {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntimeHost`.
"""
logger.info("# Distributed Agent Runtime")


host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
host.start()  # Start a host service in the background.

"""
The above code starts the host service in the background and accepts
worker connections on port 50051.

Before running worker runtimes, let's define our agent.
The agent will publish a new message on every message it receives.
It also keeps track of how many messages it has published, and 
stops publishing new messages once it has published 5 messages.
"""
logger.info("The above code starts the host service in the background and accepts")




@dataclass
class MyMessage:
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__("My agent")
        self._name = name
        self._counter = 0

    @message_handler
    async def my_message_handler(self, message: MyMessage, ctx: MessageContext) -> None:
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        logger.debug(content)
        async def run_async_code_23a5c76b():
            await self.publish_message(MyMessage(content=content), DefaultTopicId())
            return 
         = asyncio.run(run_async_code_23a5c76b())
        logger.success(format_json())

"""
Now we can set up the worker agent runtimes.
We use {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime`.
We set up two worker runtimes. Each runtime hosts one agent.
All agents publish and subscribe to the default topic, so they can see all
messages being published.

To run the agents, we publish a message from a worker.
"""
logger.info("Now we can set up the worker agent runtimes.")



worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
async def run_async_code_a6eb5afe():
    await worker1.start()
    return 
 = asyncio.run(run_async_code_a6eb5afe())
logger.success(format_json())
async def run_async_code_f4ee874a():
    await MyAgent.register(worker1, "worker1", lambda: MyAgent("worker1"))
    return 
 = asyncio.run(run_async_code_f4ee874a())
logger.success(format_json())

worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
async def run_async_code_52d4db3a():
    await worker2.start()
    return 
 = asyncio.run(run_async_code_52d4db3a())
logger.success(format_json())
async def run_async_code_0364a1f9():
    await MyAgent.register(worker2, "worker2", lambda: MyAgent("worker2"))
    return 
 = asyncio.run(run_async_code_0364a1f9())
logger.success(format_json())

async def run_async_code_e605d63a():
    await worker2.publish_message(MyMessage(content="Hello!"), DefaultTopicId())
    return 
 = asyncio.run(run_async_code_e605d63a())
logger.success(format_json())

async def run_async_code_8f25ae0d():
    await asyncio.sleep(5)
    return 
 = asyncio.run(run_async_code_8f25ae0d())
logger.success(format_json())

"""
We can see each agent published exactly 5 messages.

To stop the worker runtimes, we can call {py:meth}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime.stop`.
"""
logger.info("We can see each agent published exactly 5 messages.")

async def run_async_code_30e5da06():
    await worker1.stop()
    return 
 = asyncio.run(run_async_code_30e5da06())
logger.success(format_json())
async def run_async_code_353412df():
    await worker2.stop()
    return 
 = asyncio.run(run_async_code_353412df())
logger.success(format_json())

"""
We can call {py:meth}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntimeHost.stop`
to stop the host service.
"""
logger.info("We can call {py:meth}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntimeHost.stop`")

async def run_async_code_7b26c344():
    await host.stop()
    return 
 = asyncio.run(run_async_code_7b26c344())
logger.success(format_json())

"""
## Cross-Language Runtimes
The process described above is largely the same, however all message types MUST use shared protobuf schemas for all cross-agent message types.

## Next Steps
To see complete examples of using distributed runtime, please take a look at the following samples:

- [Distributed Workers](https://github.com/microsoft/autogen/tree/main/python/samples/core_grpc_worker_runtime)  
- [Distributed Semantic Router](https://github.com/microsoft/autogen/tree/main/python/samples/core_semantic_router)  
- [Distributed Group Chat](https://github.com/microsoft/autogen/tree/main/python/samples/core_distributed-group-chat)


"""
logger.info("## Cross-Language Runtimes")

logger.info("\n\n[DONE]", bright=True)