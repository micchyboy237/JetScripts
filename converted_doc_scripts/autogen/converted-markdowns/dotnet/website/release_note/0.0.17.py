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
# AutoGen.Net 0.0.17 Release Notes

## 🌟 What's New

1. **.NET Core Target Framework Support** ([#3203](https://github.com/microsoft/autogen/issues/3203))
   - 🚀 Added support for .NET Core to ensure compatibility and enhanced performance of AutoGen packages across different platforms.

2. **Kernel Support in Interactive Service Constructor** ([#3181](https://github.com/microsoft/autogen/issues/3181))
   - 🧠 Enhanced the Interactive Service to accept a kernel in its constructor, facilitating usage in notebook environments.

3. **Constructor Options for MLXChatAgent** ([#3126](https://github.com/microsoft/autogen/issues/3126))
   - ⚙️ Added new constructor options for `MLXChatAgent` to allow full control over chat completion flags/options.

4. **Step-by-Step Execution for Group Chat** ([#3075](https://github.com/microsoft/autogen/issues/3075))
   - 🛠️ Introduced an `IAsyncEnumerable` extension API to run group chat step-by-step, enabling developers to observe internal processes or implement early stopping mechanisms.

## 🚀 Improvements

1. **Cancellation Token Addition in Graph APIs** ([#3111](https://github.com/microsoft/autogen/issues/3111))
   - 🔄 Added cancellation tokens to async APIs in the `AutoGen.Core.Graph` class to follow best practices and enhance the control flow.

## ⚠️ API Breaking Changes

1. **FunctionDefinition Generation Stopped in Source Generator** ([#3133](https://github.com/microsoft/autogen/issues/3133))
   - 🛑 Stopped generating `FunctionDefinition` from `Azure.AI.MLX` in the source generator to eliminate unnecessary package dependencies. Migration guide:
     - ➡️ Use `ToMLXFunctionDefinition()` extension from `AutoGen.MLX` for generating `FunctionDefinition` from `AutoGen.Core.FunctionContract`.
     - ➡️ Use `FunctionContract` for metadata such as function name or parameters.

2. **Namespace Renaming for AutoGen.WebAPI** ([#3152](https://github.com/microsoft/autogen/issues/3152))
   - ✏️ Renamed the namespace of `AutoGen.WebAPI` from `AutoGen.Service` to `AutoGen.WebAPI` to maintain consistency with the project name.

3. **Semantic Kernel Version Update** ([#3118](https://github.com/microsoft/autogen/issues/3118))
   - 📈 Upgraded the Semantic Kernel version to 1.15.1 for enhanced functionality and performance improvements. This might introduce break change for those who use a lower-version semantic kernel.

## 📚 Documentation

1. **Consume AutoGen.Net Agent in AG Studio** ([#3142](https://github.com/microsoft/autogen/issues/3142))
   - Added detailed documentation on using AutoGen.Net Agent as a model in AG Studio, including examples of starting an MLX chat backend and integrating third-party MLX models.

2. **Middleware Overview Documentation Errors Fixed** ([#3129](https://github.com/microsoft/autogen/issues/3129))
   - Corrected logic and compile errors in the example code provided in the Middleware Overview documentation to ensure it runs without issues.

---

We hope you enjoy the new features and improvements in AutoGen.Net 0.0.17! If you encounter any issues or have feedback, please open a new issue on our [GitHub repository](https://github.com/microsoft/autogen/issues).
"""
logger.info("# AutoGen.Net 0.0.17 Release Notes")

logger.info("\n\n[DONE]", bright=True)