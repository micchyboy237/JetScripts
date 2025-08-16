import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for AutoGen, a framework for developing applications using AI agents
html_theme.sidebar_secondary.remove: false
sd_hide_title: true
---

<style>
.hero-title {
  font-size: 60px;
  font-weight: bold;
  margin: 2rem auto 0;
}

.wip-card {
  border: 1px solid var(--pst-color-success);
  background-color: var(--pst-color-success-bg);
  border-radius: .25rem;
  padding: 0.3rem;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 1rem;
}
</style>

# AutoGen

<div class="container">
<div class="row text-center">
<div class="col-sm-12">
<h1 class="hero-title">
AutoGen
</h1>
<h3>
A framework for building AI agents and applications
</h3>
</div>
</div>
</div>

<div style="margin-top: 2rem;">

::::{grid}
:gutter: 2

:::{grid-item-card} {fas}`palette;pst-color-primary` Studio [![PyPi autogenstudio](https://img.shields.io/badge/PyPi-autogenstudio-blue?logo=pypi)](https://pypi.org/project/autogenstudio/)
:shadow: none
:margin: 2 0 0 0
:columns: 12 12 12 12

An web-based UI for prototyping with agents without writing code.
Built on AgentChat.
"""
logger.info("# AutoGen")

pip install -U autogenstudio
autogenstudio ui --port 8080 --appdir ./myapp

"""
_Start here if you are new to AutoGen and want to prototype with agents without writing code._

+++
"""

:color: secondary

Get Started

"""
:::

:::{grid-item-card}
:shadow: none
:margin: 2 0 0 0
:columns: 12 12 12 12

<div class="sd-card-title sd-font-weight-bold docutils">

{fas}`people-group;pst-color-primary` AgentChat
[![PyPi autogen-agentchat](https://img.shields.io/badge/PyPi-autogen--agentchat-blue?logo=pypi)](https://pypi.org/project/autogen-agentchat/)

</div>
A programming framework for building conversational single and multi-agent applications.
Built on Core. Requires Python 3.10+.
"""
logger.info("A programming framework for building conversational single and multi-agent applications.")


async def main() -> None:
    agent = AssistantAgent("assistant", OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096))
    async def run_async_code_ade8da1c():
        logger.debug(await agent.run(task="Say 'Hello World!'"))
        return 
     = asyncio.run(run_async_code_ade8da1c())
    logger.success(format_json())

asyncio.run(main())

"""
_Start here if you are prototyping with agents using Python. [Migrating from AutoGen 0.2?](./user-guide/agentchat-user-guide/migration-guide.md)._

+++
"""

:color: secondary

Get Started

"""
:::

:::{grid-item-card} {fas}`cube;pst-color-primary` Core [![PyPi autogen-core](https://img.shields.io/badge/PyPi-autogen--core-blue?logo=pypi)](https://pypi.org/project/autogen-core/)
:shadow: none
:margin: 2 0 0 0
:columns: 12 12 12 12

An event-driven programming framework for building scalable multi-agent AI systems. Example scenarios:

* Deterministic and dynamic agentic workflows for business processes.
* Research on multi-agent collaboration.
* Distributed agents for multi-language applications.

_Start here if you are getting serious about building multi-agent systems._

+++
"""
logger.info("An event-driven programming framework for building scalable multi-agent AI systems. Example scenarios:")

:color: secondary

Get Started

"""
:::

:::{grid-item-card} {fas}`puzzle-piece;pst-color-primary` Extensions [![PyPi autogen-ext](https://img.shields.io/badge/PyPi-autogen--ext-blue?logo=pypi)](https://pypi.org/project/autogen-ext/)
:shadow: none
:margin: 2 0 0 0
:columns: 12 12 12 12

Implementations of Core and AgentChat components that interface with external services or other libraries.
You can find and use community extensions or create your own. Examples of built-in extensions:

* {py:class}`~autogen_ext.tools.mcp.McpWorkbench` for using Model-Context Protocol (MCP) servers.
* {py:class}`~autogen_ext.agents.openai.OllamaAssistantAgent` for using Assistant API.
* {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` for running model-generated code in a Docker container.
* {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime` for distributed agents.

+++

<a class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-secondary reference internal" href="user-guide/extensions-user-guide/discover.html"><span class="doc">Discover Community Extensions</span></a>
<a class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-secondary reference internal" href="user-guide/extensions-user-guide/create-your-own.html"><span class="doc">Create New Extension</span></a>

:::

::::

</div>
"""
logger.info("Implementations of Core and AgentChat components that interface with external services or other libraries.")

:maxdepth: 3
:hidden:

user-guide/agentchat-user-guide/index
user-guide/core-user-guide/index
user-guide/extensions-user-guide/index
Studio <user-guide/autogenstudio-user-guide/index>
reference/index

logger.info("\n\n[DONE]", bright=True)