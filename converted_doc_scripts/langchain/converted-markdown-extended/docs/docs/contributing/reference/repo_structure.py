from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar_position: 0.5
---
# Repository Structure

If you plan on contributing to LangChain code or documentation, it can be useful
to understand the high level structure of the repository.

LangChain is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.
You can check out our [installation guide](/docs/how_to/installation/) for more on how they fit together.

Here's the structure visualized as a tree:

.
├── cookbook # Tutorials and examples
├── docs # Contains content for the documentation here: https://python.langchain.com/
├── libs
│   ├── langchain
│   │   ├── langchain
│   │   ├── tests/unit_tests # Unit tests (present in each package not shown for brevity)
│   │   ├── tests/integration_tests # Integration tests (present in each package not shown for brevity)
│   ├── community # Third-party integrations
│   │   ├── langchain-community
│   ├── core # Base interfaces for key abstractions
│   │   ├── langchain-core
│   ├── experimental # Experimental components and chains
│   │   ├── langchain-experimental
|   ├── cli # Command line interface
│   │   ├── langchain-cli
│   ├── text-splitters
│   │   ├── langchain-text-splitters
│   ├── standard-tests
│   │   ├── langchain-standard-tests
│   ├── partners
│       ├── langchain-partner-1
│       ├── langchain-partner-2
│       ├── ...
│
├── templates # A collection of easily deployable reference architectures for a wide variety of tasks.

The root directory also contains the following files:

* `pyproject.toml`: Dependencies for building docs and linting docs, cookbook.
* `Makefile`: A file that contains shortcuts for building, linting and docs and cookbook.

There are other files in the root directory level, but their presence should be self-explanatory. Feel free to browse around!

## Documentation

The `/docs` directory contains the content for the documentation that is shown
at [python.langchain.com](https://python.langchain.com/) and the associated [API Reference](https://python.langchain.com/api_reference/).

See the [documentation](../how_to/documentation/index.mdx) guidelines to learn how to contribute to the documentation.

## Code

The `/libs` directory contains the code for the LangChain packages.

To learn more about how to contribute code see the following guidelines:

- [Code](../how_to/code/index.mdx): Learn how to develop in the LangChain codebase.
- [Integrations](../how_to/integrations/index.mdx): Learn how to contribute to third-party integrations to `langchain-community` or to start a new partner package.
- [Testing](../how_to/testing.mdx): Guidelines to learn how to write tests for the packages.
"""
logger.info("# Repository Structure")

logger.info("\n\n[DONE]", bright=True)