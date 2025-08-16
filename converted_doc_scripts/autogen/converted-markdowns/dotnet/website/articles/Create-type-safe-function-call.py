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
## Create type-safe function call using AutoGen.SourceGenerator

`AutoGen` provides a source generator to easness the trouble of manually craft function definition and function call wrapper from a function. To use this feature, simply add the `AutoGen.SourceGenerator` package to your project and decorate your function with @AutoGen.Core.FunctionAttribute.
"""
logger.info("## Create type-safe function call using AutoGen.SourceGenerator")

dotnet add package AutoGen.SourceGenerator

"""
> [!NOTE]
> It's recommended to enable structural xml document support by setting `GenerateDocumentationFile` property to true in your project file. This allows source generator to leverage the documentation of the function when generating the function definition.
"""

<PropertyGroup>
    <!-- This enables structural xml document support -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
</PropertyGroup>

"""
Then, create a `public partial` class to host the methods you want to use in AutoGen agents. The method has to be a `public` instance method and its return type must be `Task<string>`. After the methods is defined, mark them with @AutoGen.FunctionAttribute attribute:

> [!NOTE]
> A `public partial` class is required for the source generator to generate code.
> The method has to be a `public` instance method and its return type must be `Task<string>`.
> Mark the method with @AutoGen.Core.FunctionAttribute attribute.

Firstly, import the required namespaces:

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/TypeSafeFunctionCallCodeSnippet.cs?name=weather_report_using_statement)]

Then, create a `WeatherReport` function and mark it with @AutoGen.Core.FunctionAttribute:

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/TypeSafeFunctionCallCodeSnippet.cs?name=weather_report)]

The source generator will generate the @AutoGen.Core.FunctionContract and function call wrapper for `WeatherReport` in another partial class based on its signature and structural comments. The @AutoGen.Core.FunctionContract is introduced by [#1736](https://github.com/microsoft/autogen/pull/1736) and contains all the necessary metadata such as function name, parameters, and return type. It is LLM independent and can be used to generate openai function definition or semantic kernel function. The function call wrapper is a helper class that provides a type-safe way to call the function.

> [!NOTE]
> If you are using VSCode as your editor, you may need to restart the editor to see the generated code.

The following code shows how to generate openai function definition from the @AutoGen.Core.FunctionContract and call the function using the function call wrapper.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/TypeSafeFunctionCallCodeSnippet.cs?name=weather_report_consume)]
"""
logger.info("Then, create a `public partial` class to host the methods you want to use in AutoGen agents. The method has to be a `public` instance method and its return type must be `Task<string>`. After the methods is defined, mark them with @AutoGen.FunctionAttribute attribute:")

logger.info("\n\n[DONE]", bright=True)