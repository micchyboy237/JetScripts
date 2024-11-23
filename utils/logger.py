import logging
from typing import List, Callable, TypedDict, Any, Optional

# ANSI color codes
BOLD = "\u001b[1m"
PINK = BOLD + "\u001b[38;5;201m"
BRIGHT_PINK = BOLD + "\u001b[48;5;201m"
CYAN = BOLD + "\u001b[38;5;45m"
BRIGHT_CYAN = BOLD + "\u001b[48;5;45m"
YELLOW = BOLD + "\u001b[38;5;220m"
BRIGHT_YELLOW = BOLD + "\u001b[48;5;220m"
RED = BOLD + "\u001b[38;5;196m"
BRIGHT_RED = BOLD + "\u001b[48;5;196m"
GREEN = BOLD + "\u001b[38;5;40m"
BRIGHT_GREEN = BOLD + "\u001b[48;5;40m"
WHITE = BOLD + "\u001b[38;5;15m"
GRAY = "\u001b[38;5;250m"
RESET = "\u001b[0m"

COLORS = {
    "INFO": PINK,
    "BRIGHT_INFO": BRIGHT_PINK,
    "DEBUG": CYAN,
    "BRIGHT_DEBUG": BRIGHT_CYAN,
    "WARNING": YELLOW,
    "BRIGHT_WARNING": BRIGHT_YELLOW,
    "ERROR": RED,
    "BRIGHT_ERROR": BRIGHT_RED,
    "SUCCESS": GREEN,
    "BRIGHT_SUCCESS": BRIGHT_GREEN,
    "LOG": WHITE,
    "GRAY": GRAY,
}


class LoggerMethods(TypedDict):
    info: Callable[[str, Optional[bool]], Optional[bool]]
    debug: Callable[[str, Optional[bool]], Optional[bool]]
    error: Callable[[str, Optional[bool]], Optional[bool]]
    warning: Callable[[str, Optional[bool]], Optional[bool]]
    success: Callable[[str, Optional[bool]], Optional[bool]]
    log: Callable[[str, Optional[bool]], Optional[bool]]


class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def custom_logger_method(self, level: str) -> Callable[[str, Optional[bool]], None]:
        def wrapper(message: str, bright: bool = False, *args: Any, **kwargs: Any) -> None:
            actual_level = f"BRIGHT_{level}" if bright else level
            colored_message = f"{COLORS[actual_level]}{message}{RESET}"

            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(colored_message)
        return wrapper

    def log(self, *messages: str, colors: Optional[List[str]] = None) -> None:
        if colors is None:
            colors = ["LOG"] * len(messages)
        else:
            colors = colors * \
                ((len(messages) + len(colors) - 1) // len(colors))

        formatted_messages = [
            f"{COLORS[color]}{message}{RESET}" for message, color in zip(messages, colors)
        ]

        output = " ".join(formatted_messages)
        self.logger.info(output)

    def __getattr__(self, name: str) -> Callable[[str, Optional[bool]], None]:
        if name.upper() in ["INFO", "DEBUG", "ERROR", "WARNING", "SUCCESS", "LOG"]:
            return self.custom_logger_method(name.upper())
        raise AttributeError(
            f"'CustomLogger' object has no attribute '{name}'")


# Initialize the logger
logger = CustomLogger()

# # Test cases
# logger.log("\n==== LOGGER METHODS =====")
# logger.info("This is an info message.")
# logger.info("This is a bright info message.", bright=True)
# logger.debug("This is a debug message.")
# logger.debug("This is a bright debug message.", bright=True)
# logger.warning("This is a warning message.")
# logger.warning("This is a bright warning message.", bright=True)
# logger.error("This is an error message.")
# logger.error("This is a bright error message.", bright=True)
# logger.success("This is a success message.")
# logger.success("This is a bright success message.", bright=True)
# logger.log("This is a default log message.")
# logger.log(
#     "2 multi-color: Message 1",
#     "Message 2",
#     colors=["LOG", "DEBUG"]
# )
# logger.log(
#     "3 multi-color: Message 1",
#     "Message 2",
#     "Message 3",
#     colors=["GRAY", "DEBUG", "INFO"]
# )
# logger.log("====== END LOGGER METHODS ======\n")
