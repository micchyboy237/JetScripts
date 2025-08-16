from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# pyautogen

> **NOTE:** This is a proxy package for the latest version of [`autogen-agentchat`](https://pypi.org/project/autogen-agentchat/). If you are looking for the 0.2.x version, please pin to `pyautogen~=0.2.0`.
> To migrate from 0.2.x to the latest version, please refer to the [migration guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html).
> Read our [previous clarification regarding to forks](https://github.com/microsoft/autogen/discussions/4217).
> We have regained admin access to this package.

AutoGen is a framework for creating multi-agent AI applications that can act autonomously or work alongside humans.

- [Project homepage](https://github.com/microsoft/autogen)
- [Documentation](https://microsoft.github.io/autogen/)
- [Discord](https://aka.ms/autogen-discord)
- [Contact](mailto:autogen@microsoft.com)
"""
logger.info("# pyautogen")

logger.info("\n\n[DONE]", bright=True)