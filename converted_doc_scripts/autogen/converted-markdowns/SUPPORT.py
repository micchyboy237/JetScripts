from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Support

## How to file issues and get help

This project uses [GitHub Issues](https://github.com/microsoft/autogen/issues)
to track bugs and feature requests. Please search the existing
issues before filing new issues to avoid duplicates.  For new issues, file your bug or
feature request as a new Issue.

For help and questions about using this project, please use
[GitHub Discussion](https://github.com/microsoft/autogen/discussions).
Follow [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
when participating in the forum.

## Microsoft Support Policy

Support for this project is limited to the resources listed above.
"""
logger.info("# Support")

logger.info("\n\n[DONE]", bright=True)