from jet.logger import logger
import CodeBlock from "@theme/CodeBlock";
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';
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
sidebar_position: 2
---

# How to install LangChain packages

The LangChain ecosystem is split into different packages, which allow you to choose exactly which pieces of
functionality to install.

## Official release

To install the main `langchain` package, run:


<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install langchain</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>

While this package acts as a sane starting point to using LangChain,
much of the value of LangChain comes when integrating it with various model providers, datastores, etc.
By default, the dependencies needed to do that are NOT installed. You will need to install the dependencies for specific integrations separately, which we show below.

## Ecosystem packages

With the exception of the `langsmith` SDK, all packages in the LangChain ecosystem depend on `langchain-core`, which contains base
classes and abstractions that other packages use. The dependency graph below shows how the different packages are related.
A directed arrow indicates that the source package depends on the target package:

![](/img/ecosystem_packages.png)

When installing a package, you do not need to explicitly install that package's explicit dependencies (such as `langchain-core`).
However, you may choose to if you are using a feature only available in a certain version of that dependency.
If you do, you should make sure that the installed or pinned version is compatible with any other integration packages you use.

### LangChain core
The `langchain-core` package contains base abstractions that the rest of the LangChain ecosystem uses, along with the LangChain Expression Language. It is automatically installed by `langchain`, but can also be used separately. Install with:
"""
logger.info("# How to install LangChain packages")

pip install langchain-core

"""
### Integration packages

Certain integrations like Ollama and Ollama have their own packages.
Any integrations that require their own package will be documented as such in the [Integration docs](/docs/integrations/providers/).
You can see a list of all integration packages in the [API reference](https://python.langchain.com/api_reference/) under the "Partner libs" dropdown.
To install one of these run:
"""
logger.info("### Integration packages")

pip install langchain-ollama

"""
Any integrations that haven't been split out into their own packages will live in the `langchain-community` package. Install with:
"""
logger.info("Any integrations that haven't been split out into their own packages will live in the `langchain-community` package. Install with:")

pip install langchain-community

"""
### LangChain experimental
The `langchain-experimental` package holds experimental LangChain code, intended for research and experimental uses.
Install with:
"""
logger.info("### LangChain experimental")

pip install langchain-experimental

"""
### LangGraph
`langgraph` is a library for building stateful, multi-actor applications with LLMs. It integrates smoothly with LangChain, but can be used without it.
Install with:
"""
logger.info("### LangGraph")

pip install langgraph

"""
### LangServe
LangServe helps developers deploy LangChain runnables and chains as a REST API.
LangServe is automatically installed by LangChain CLI.
If not using LangChain CLI, install with:
"""
logger.info("### LangServe")

pip install "langserve[all]"

"""
for both client and server dependencies. Or `pip install "langserve[client]"` for client code, and `pip install "langserve[server]"` for server code.

### LangChain CLI
The LangChain CLI is useful for working with LangChain templates and other LangServe projects.
Install with:
"""
logger.info("### LangChain CLI")

pip install langchain-cli

"""
### LangSmith SDK
The LangSmith SDK is automatically installed by LangChain. However, it does not depend on
`langchain-core`, and can be installed and used independently if desired.
If you are not using LangChain, you can install it with:
"""
logger.info("### LangSmith SDK")

pip install langsmith

"""
### From source

If you want to install a package from source, you can do so by cloning the [main LangChain repo](https://github.com/langchain-ai/langchain), enter the directory of the package you want to install `PATH/TO/REPO/langchain/libs/{package}`, and run:
"""
logger.info("### From source")

pip install -e .

"""
LangGraph, LangSmith SDK, and certain integration packages live outside the main LangChain repo. You can see [all repos here](https://github.com/langchain-ai).
"""
logger.info("LangGraph, LangSmith SDK, and certain integration packages live outside the main LangChain repo. You can see [all repos here](https://github.com/langchain-ai).")

logger.info("\n\n[DONE]", bright=True)