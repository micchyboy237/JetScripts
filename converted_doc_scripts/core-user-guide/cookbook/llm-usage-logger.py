from autogen_core import EVENT_LOGGER_NAME
from autogen_core.logging import LLMCallEvent
from jet.logger import CustomLogger
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Tracking LLM usage with a logger

The model clients included in AutoGen emit structured events that can be used to track the usage of the model. This notebook demonstrates how to use the logger to track the usage of the model.

These events are logged to the logger with the name: :py:attr:`autogen_core.EVENT_LOGGER_NAME`.
"""
logger.info("# Tracking LLM usage with a logger")




class LLMUsageTracker(logging.Handler):
    def __init__(self) -> None:
        """Logging handler that tracks the number of tokens used in the prompt and completion."""
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record. To be used by the logging module."""
        try:
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens
        except Exception:
            self.handleError(record)

"""
Then, this logger can be attached like any other Python logger and the values read after the model is run.
"""
logger.info("Then, this logger can be attached like any other Python logger and the values read after the model is run.")


logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.INFO)
llm_usage = LLMUsageTracker()
logger.handlers = [llm_usage]


logger.debug(llm_usage.prompt_tokens)
logger.debug(llm_usage.completion_tokens)

logger.info("\n\n[DONE]", bright=True)