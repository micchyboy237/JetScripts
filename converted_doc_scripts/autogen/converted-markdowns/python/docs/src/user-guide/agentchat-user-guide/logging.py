from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
import logging

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Logging

AutoGen uses Python's built-in [`logging`](https://docs.python.org/3/library/logging.html) module.

To enable logging for AgentChat, you can use the following code:
"""
logger.info("# Logging")



logging.basicConfig(level=logging.WARNING)

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
trace_logger.addHandler(logging.StreamHandler())
trace_logger.setLevel(logging.DEBUG)

event_logger = logging.getLogger(EVENT_LOGGER_NAME)
event_logger.addHandler(logging.StreamHandler())
event_logger.setLevel(logging.DEBUG)

"""
To enable additional logs such as model client calls and agent runtime events,
please refer to the [Core Logging Guide](../core-user-guide/framework/logging.md).
"""
logger.info("To enable additional logs such as model client calls and agent runtime events,")

logger.info("\n\n[DONE]", bright=True)