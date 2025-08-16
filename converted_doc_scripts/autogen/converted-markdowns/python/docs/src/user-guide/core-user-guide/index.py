

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
      User Guide for AutoGen Core, a framework for building multi-agent applications with AI agents.
---

# Core
"""
logger.info("# Core")

:maxdepth: 1
:hidden:

installation
quickstart

"""

"""

:maxdepth: 1
:hidden:
:caption: Core Concepts

core-concepts/agent-and-multi-agent-application
core-concepts/architecture
core-concepts/application-stack
core-concepts/agent-identity-and-lifecycle
core-concepts/topic-and-subscription

"""

"""

:maxdepth: 1
:hidden:
:caption: Framework Guide

framework/agent-and-agent-runtime
framework/message-and-communication
framework/logging
framework/telemetry
framework/distributed-agent-runtime
framework/component-config

"""

"""

:maxdepth: 1
:hidden:
:caption: Components Guide

components/model-clients
components/model-context
components/tools
components/workbench
components/command-line-code-executors

"""

"""

:maxdepth: 1
:hidden:
:caption: Multi-Agent Design Patterns

design-patterns/intro
design-patterns/concurrent-agents
design-patterns/sequential-workflow
design-patterns/group-chat
design-patterns/handoffs
design-patterns/mixture-of-agents
design-patterns/multi-agent-debate
design-patterns/reflection
design-patterns/code-execution-groupchat

"""

"""

:maxdepth: 1
:hidden:
:caption: More

cookbook/index
faqs

"""
AutoGen core offers an easy way to quickly build event-driven, distributed, scalable, resilient AI agent systems. Agents are developed by using the [Actor model](https://en.wikipedia.org/wiki/Actor_model). You can build and run your agent system locally and easily move to a distributed system in the cloud when you are ready.

Key features of AutoGen core include:
"""
logger.info("AutoGen core offers an easy way to quickly build event-driven, distributed, scalable, resilient AI agent systems. Agents are developed by using the [Actor model](https://en.wikipedia.org/wiki/Actor_model). You can build and run your agent system locally and easily move to a distributed system in the cloud when you are ready.")

:grid-columns: 1 2 2 3

- header: "{fas}`network-wired;pst-color-primary` Asynchronous Messaging"
  content: "Agents communicate through asynchronous messages, enabling event-driven and request/response communication models."
- header: "{fas}`cube;pst-color-primary` Scalable & Distributed"
  content: "Enable complex scenarios with networks of agents across organizational boundaries."
- header: "{fas}`code;pst-color-primary` Multi-Language Support"
  content: "Python & Dotnet interoperating agents today, with more languages coming soon."
- header: "{fas}`globe;pst-color-primary` Modular & Extensible"
  content: "Highly customizable with features like custom agents, memory as a service, tools registry, and model library."
- header: "{fas}`puzzle-piece;pst-color-primary` Observable & Debuggable"
  content: "Easily trace and debug your agent systems."
- header: "{fas}`project-diagram;pst-color-primary` Event-Driven Architecture"
  content: "Build event-driven, distributed, scalable, and resilient AI agent systems."

logger.info("\n\n[DONE]", bright=True)