

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen AgentChat

- [Documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)

AgentChat is a high-level API for building multi-agent applications.
It is built on top of the [`autogen-core`](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html) package.
For beginner users, AgentChat is the recommended starting point.
For advanced users, [`autogen-core`](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html)'s event-driven
programming model provides more flexibility and control over the underlying components.

AgentChat provides intuitive defaults, such as **Agents** with preset
behaviors and **Teams** with predefined [multi-agent design patterns](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/intro.html).
"""
logger.info("# AutoGen AgentChat")

logger.info("\n\n[DONE]", bright=True)