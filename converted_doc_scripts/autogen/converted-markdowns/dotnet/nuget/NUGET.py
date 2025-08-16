from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
### About AutoGen for .NET
`AutoGen for .NET` is the official .NET SDK for [AutoGen](https://github.com/microsoft/autogen). It enables you to create LLM agents and construct multi-agent workflows with ease. It also provides integration with popular platforms like Ollama, Semantic Kernel, and LM Studio.

### Gettings started
- Find documents and examples on our [document site](https://microsoft.github.io/autogen-for-net/)
- Report a bug or request a feature by creating a new issue in our [github repo](https://github.com/microsoft/autogen)
- Consume the nightly build package from one of the [nightly build feeds](https://microsoft.github.io/autogen-for-net/articles/Installation.html#nighly-build)
"""
logger.info("### About AutoGen for .NET")

logger.info("\n\n[DONE]", bright=True)