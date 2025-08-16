import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## AutoGen.LMStudio

This package provides support for consuming openai-like API from LMStudio local server.

## Installation
To use `AutoGen.LMStudio`, add the following package to your `.csproj` file:
"""
logger.info("## AutoGen.LMStudio")

<ItemGroup>
    <PackageReference Include="AutoGen.LMStudio" Version="AUTOGEN_VERSION" />
</ItemGroup>

"""
## Usage
"""
logger.info("## Usage")

using AutoGen.LMStudio;
var localServerEndpoint = "localhost";
var port = 5000;
var lmStudioConfig = new LMStudioConfig(localServerEndpoint, port);
var agent = new LMStudioAgent(
    name: "agent",
    systemMessage: "You are an agent that help user to do some tasks.",
    lmStudioConfig: lmStudioConfig)
    .RegisterPrintMessage(); // register a hook to print message nicely to console

async def run_async_code_03aac235():
    await agent.SendAsync("Can you write a piece of C# code to calculate 100th of fibonacci?");
    return 
 = asyncio.run(run_async_code_03aac235())
logger.success(format_json())

"""
## Update history
### Update on 0.0.7 (2024-02-11)
- Add `LMStudioAgent` to support consuming openai-like API from LMStudio local server.
"""
logger.info("## Update history")

logger.info("\n\n[DONE]", bright=True)