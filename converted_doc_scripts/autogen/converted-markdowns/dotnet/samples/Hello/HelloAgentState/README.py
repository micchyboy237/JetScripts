import asyncio
from jet.transformers.formatters import format_json
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
# AutoGen 0.4 .NET Hello World Sample

This [sample](Program.cs) demonstrates how to create a simple .NET console application that listens for an event and then orchestrates a series of actions in response.

## Prerequisites

To run this sample, you'll need: [.NET 8.0](https://dotnet.microsoft.com/en-us/) or later.
Also recommended is the [GitHub CLI](https://cli.github.com/).

## Instructions to run the sample
"""
logger.info("# AutoGen 0.4 .NET Hello World Sample")

gh repo clone microsoft/autogen
cd dotnet/samples/Hello
dotnet run

"""
## Key Concepts

This sample illustrates how to create your own agent that inherits from a base agent and listens for an event. It also shows how to use the SDK's App Runtime locally to start the agent and send messages.

Flow Diagram:
"""
logger.info("## Key Concepts")

%%{init: {'theme':'forest'}}%%
graph LR;
    A[Main] --> |"PublishEventAsync(NewMessage('World'))"| B{"Handle(NewMessageReceived item, CancellationToken cancellationToken = default)"}
    B --> |"PublishEventAsync(Output('***Hello, World***'))"| C[ConsoleAgent]
    C --> D{"WriteConsole()"}
    B --> |"PublishEventAsync(ConversationClosed('Goodbye'))"| E{"Handle(ConversationClosed item, CancellationToken cancellationToken = default)"}
    B --> |"PublishEventAsync(Output('***Goodbye***'))"| C
    E --> F{"Shutdown()"}

"""
### Writing Event Handlers

The heart of an autogen application are the event handlers. Agents select a ```TopicSubscription``` to listen for events on a specific topic. When an event is received, the agent's event handler is called with the event data.

Within that event handler you may optionally *emit* new events, which are then sent to the event bus for other agents to process. The EventTypes are declared gRPC ProtoBuf messages that are used to define the schema of the event.  The default protos are available via the ```Microsoft.AutoGen.Contracts;``` namespace and are defined in [autogen/protos](/autogen/protos). The EventTypes are registered in the agent's constructor using the ```IHandle``` interface.
"""
logger.info("### Writing Event Handlers")

TopicSubscription("HelloAgents")]
public class HelloAgent(
    iAgentWorker worker,
    [FromKeyedServices("AgentsMetadata")] AgentsMetadata typeRegistry) : ConsoleAgent(
        worker,
        typeRegistry),
        ISayHello,
        IHandle<NewMessageReceived>,
        IHandle<ConversationClosed>
{
    public async Task Handle(NewMessageReceived item, CancellationToken cancellationToken = default)
    {
        async def run_async_code_6b5b7e6f():
            async def run_async_code_0529f29f():
                var response = await SayHello(item.Message).ConfigureAwait(false);
                return var response
            var response = asyncio.run(run_async_code_0529f29f())
            logger.success(format_json(var response))
            return var response
        var response = asyncio.run(run_async_code_6b5b7e6f())
        logger.success(format_json(var response))
        var evt = new Output
        {
            Message = response
        }.ToCloudEvent(this.AgentId.Key);
        async def run_async_code_c568eddd():
            await PublishEventAsync(evt).ConfigureAwait(false);
            return 
         = asyncio.run(run_async_code_c568eddd())
        logger.success(format_json())
        var goodbye = new ConversationClosed
        {
            UserId = this.AgentId.Key,
            UserMessage = "Goodbye"
        }.ToCloudEvent(this.AgentId.Key);
        async def run_async_code_24d8f771():
            await PublishEventAsync(goodbye).ConfigureAwait(false);
            return 
         = asyncio.run(run_async_code_24d8f771())
        logger.success(format_json())
    }

"""
### Inheritance and Composition

This sample also illustrates inheritance in AutoGen. The `HelloAgent` class inherits from `ConsoleAgent`, which is a base class that provides a `WriteConsole` method.

### Starting the Application Runtime

AuotoGen provides a flexible runtime ```Microsoft.AutoGen.Agents.App``` that can be started in a variety of ways. The `Program.cs` file demonstrates how to start the runtime locally and send a message to the agent all in one go using the ```App.PublishMessageAsync``` method.
"""
logger.info("### Inheritance and Composition")

// send a message to the agent
async def run_async_code_936e4c88():
    var app = await App.PublishMessageAsync("HelloAgents", new NewMessageReceived
    return var app
var app = asyncio.run(run_async_code_936e4c88())
logger.success(format_json(var app))
{
    Message = "World"
}, local: true);

async def run_async_code_2b2e16bd():
    await App.RuntimeApp!.WaitForShutdownAsync();
    return 
 = asyncio.run(run_async_code_2b2e16bd())
logger.success(format_json())
async def run_async_code_8fd394bf():
    await app.WaitForShutdownAsync();
    return 
 = asyncio.run(run_async_code_8fd394bf())
logger.success(format_json())

"""
### Sending Messages

The set of possible Messages is defined in gRPC ProtoBuf specs. These are then turned into C# classes by the gRPC tools. You can define your own Message types by creating a new .proto file in your project and including the gRPC tools in your ```.csproj``` file:
"""
logger.info("### Sending Messages")

syntax = "proto3";
package devteam;
option csharp_namespace = "DevTeam.Shared";
message NewAsk {
  string org = 1;
  string repo = 2;
  string ask = 3;
  int64 issue_number = 4;
}
message ReadmeRequested {
   string org = 1;
   string repo = 2;
   int64 issue_number = 3;
   string ask = 4;
}

"""

"""

<ItemGroup>
    <PackageReference Include="Google.Protobuf" />
    <PackageReference Include="Grpc.Tools" PrivateAssets="All" />
    <Protobuf Include="..\Protos\messages.proto" Link="Protos\messages.proto" />
  </ItemGroup>

"""
You can send messages using the [```Microsoft.AutoGen.Agents.AgentWorker``` class](autogen/dotnet/src/Microsoft.AutoGen/Agents/AgentWorker.cs). Messages are wrapped in [the CloudEvents specification](https://cloudevents.io) and sent to the event bus.

### Managing State

There is a simple API for persisting agent state.
"""
logger.info("### Managing State")

async def run_async_code_b9dff7bd():
    await Store(new AgentState
    return 
 = asyncio.run(run_async_code_b9dff7bd())
logger.success(format_json())
            {
                AgentId = this.AgentId,
                TextData = entry
            }).ConfigureAwait(false);

"""
which can be read back using Read:
"""
logger.info("which can be read back using Read:")

async def run_async_code_78d8ad5f():
    async def run_async_code_3e5dee20():
        State = await Read<AgentState>(this.AgentId).ConfigureAwait(false);
        return State
    State = asyncio.run(run_async_code_3e5dee20())
    logger.success(format_json(State))
    return State
State = asyncio.run(run_async_code_78d8ad5f())
logger.success(format_json(State))

logger.info("\n\n[DONE]", bright=True)