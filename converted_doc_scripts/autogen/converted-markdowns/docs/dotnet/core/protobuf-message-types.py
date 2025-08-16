

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Protocol Buffers to Define Message Types

For a message to be sent using a runtime other than the @Microsoft.AutoGen.Core.InProcessRuntime, it must be defined as a Protocol Buffers message. This is because the message is serialized and deserialized using Protocol Buffers. This requirement may be relaxed in future by allowing for converters, custom serialization, or other mechanisms.

## How to include Protocol Buffers in a .NET project

The .proto file which defines the message types must be included in the project, which will automatically generate the C# classes for the messages.

1. Include `Grpc.Tools` package in your `.csproj` file:
"""
logger.info("# Using Protocol Buffers to Define Message Types")

<PackageReference Include="Grpc.Tools" PrivateAssets="All" />

"""
2. Create an include a `.proto` file in the project:
"""
logger.info("2. Create an include a `.proto` file in the project:")

<ItemGroup>
  <Protobuf Include="messages.proto" GrpcServices="Client;Server" Link="messages.proto" />
</ItemGroup>

"""
3. define your messages as specified in the [Protocol Buffers Language Guide](https://protobuf.dev/programming-guides/proto3/)
"""
logger.info("3. define your messages as specified in the [Protocol Buffers Language Guide](https://protobuf.dev/programming-guides/proto3/)")

syntax = "proto3";

package HelloAgents;

option csharp_namespace = "MyAgentsProtocol";

message TextMessage {
    string Source = 1;
    string Content = 2;
}

"""
4. Code against the generated class for handling, sending and publishing messages:
"""
logger.info("4. Code against the generated class for handling, sending and publishing messages:")

using Microsoft.AutoGen.Contracts;
using Microsoft.AutoGen.Core;
using MyAgentsProtocol;

[TypeSubscription("default")]
public class Checker(
    AgentId id,
    IAgentRuntime runtime,
    ) :
        BaseAgent(id, runtime, "MyAgent", null),
        IHandle<TextMessage>
{
    public async ValueTask HandleAsync(TextMessage item, MessageContext messageContext)
    {
        Console.WriteLine($"Received message from {item.Source}: {item.Content}");
    }
}

logger.info("\n\n[DONE]", bright=True)