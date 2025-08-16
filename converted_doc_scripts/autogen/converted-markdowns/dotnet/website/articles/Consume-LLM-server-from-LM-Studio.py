

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Consume LLM server from LM Studio
You can use @AutoGen.LMStudio.LMStudioAgent from `AutoGen.LMStudio` package to consume openai-like API from LMStudio local server.

### What's LM Studio
[LM Studio](https://lmstudio.ai/) is an app that allows you to deploy and inference hundreds of thousands of open-source language model on your local machine. It provides an in-app chat ui plus an openai-like API to interact with the language model programmatically.

### Installation
- Install LM studio if you haven't done so. You can find the installation guide [here](https://lmstudio.ai/)
- Add `AutoGen.LMStudio` to your project.
"""
logger.info("## Consume LLM server from LM Studio")

<ItemGroup>
    <PackageReference Include="AutoGen.LMStudio" Version="AUTOGEN_LMSTUDIO_VERSION" />
</ItemGroup>

"""
### Usage
The following code shows how to use `LMStudioAgent` to write a piece of C# code to calculate 100th of fibonacci. Before running the code, make sure you have local server from LM Studio running on `localhost:1234`.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example08_LMStudio.cs?name=lmstudio_using_statements)]
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example08_LMStudio.cs?name=lmstudio_example_1)]
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)