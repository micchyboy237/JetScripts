from autogen_core import EVENT_LOGGER_NAME
from autogen_core import TRACE_LOGGER_NAME
from dataclasses import dataclass
from jet.logger import CustomLogger
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Logging

AutoGen uses Python's built-in [`logging`](https://docs.python.org/3/library/logging.html) module.

There are two kinds of logging:

- **Trace logging**: This is used for debugging and is human readable messages to indicate what is going on. This is intended for a developer to understand what is happening in the code. The content and format of these logs should not be depended on by other systems.
  - Name: {py:attr}`~autogen_core.TRACE_LOGGER_NAME`.
- **Structured logging**: This logger emits structured events that can be consumed by other systems. The content and format of these logs can be depended on by other systems.
  - Name: {py:attr}`~autogen_core.EVENT_LOGGER_NAME`.
  - See the module {py:mod}`autogen_core.logging` to see the available events.
- {py:attr}`~autogen_core.ROOT_LOGGER_NAME` can be used to enable or disable all logs.

## Enabling logging output

To enable trace logging, you can use the following code:
"""
logger.info("# Logging")



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

"""
To enable structured logging, you can use the following code:
"""
logger.info("To enable structured logging, you can use the following code:")



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

"""
### Structured logging

Structured logging allows you to write handling logic that deals with the actual events including all fields rather than just a formatted string.

For example, if you had defined this custom event and were emitting it. Then you could write the following handler to receive it.
"""
logger.info("### Structured logging")


@dataclass
class MyEvent:
    timestamp: str
    message: str

class MyHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, MyEvent):
                logger.debug(f"Timestamp: {record.msg.timestamp}, Message: {record.msg.message}")
        except Exception:
            self.handleError(record)

"""
And this is how you could use it:
"""
logger.info("And this is how you could use it:")

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.INFO)
my_handler = MyHandler()
logger.handlers = [my_handler]

"""
## Emitting logs

These two names are the root loggers for these types. Code that emits logs should use a child logger of these loggers. For example, if you are writing a module `my_module` and you want to emit trace logs, you should use the logger named:
"""
logger.info("## Emitting logs")


logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.my_module")

"""
### Emitting structured logs

If your event is a dataclass, then it could be emitted in code like this:
"""
logger.info("### Emitting structured logs")


@dataclass
class MyEvent:
    timestamp: str
    message: str

logger = logging.getLogger(EVENT_LOGGER_NAME + ".my_module")
logger.info(MyEvent("timestamp", "message"))

logger.info("\n\n[DONE]", bright=True)