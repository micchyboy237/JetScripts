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
### Get start with AutoGen for dotnet
[![dotnet-ci](https://github.com/autogenhub/autogen/actions/workflows/dotnet-build.yml/badge.svg)](https://github.com/autogenhub/autogen/actions/workflows/dotnet-build.yml)
[![Discord](https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat)](https://discord.gg/pAbnFJrkgZ)
[![NuGet version](https://badge.fury.io/nu/AutoGen.Core.svg)](https://badge.fury.io/nu/AutoGen.Core)

Firstly, add `AutoGen` package to your project.
"""
logger.info("### Get start with AutoGen for dotnet")

dotnet add package AutoGen

"""
> [!NOTE]
> For more information about installing packages, please check out the [installation guide](Installation.md).

Then you can start with the following code snippet to create a conversable agent and chat with it.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/GetStartCodeSnippet.cs?name=snippet_GetStartCodeSnippet)]
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/GetStartCodeSnippet.cs?name=code_snippet_1)]

### Tutorial
Getting started with AutoGen.Net by following the [tutorial](../tutorial/Chat-with-an-agent.md) series.
### Examples
You can find more examples under the [sample project](https://github.com/autogenhub/autogen/tree/dotnet/dotnet/sample/AutoGen.BasicSamples).

### Report a bug or request a feature
You can report a bug or request a feature by creating a new issue in the [github issue](https://github.com/autogenhub/autogen/issues) and specifying label the label "donet"
"""
logger.info("### Tutorial")

logger.info("\n\n[DONE]", bright=True)