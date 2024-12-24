import json
from jet.executor import run_commands
from jet.logger import logger


def main_run_commands(commands: list[str], work_dir=None):
    """Test the run_commands function."""

    command_results = {}

    current_command = ""
    current_command_result = ""
    for output in run_commands(commands, work_dir=work_dir):
        if output.startswith("command:"):
            command = output.split("command:")[1].strip()

            if current_command != command:
                # Store current command result
                if current_command_result:
                    command_results[current_command] = current_command_result.strip(
                    )
                    logger.log("Command:", current_command,
                               colors=["GRAY", "INFO"])
                    logger.log("Result:", current_command_result,
                               colors=["GRAY", "SUCCESS"])

                # Reset for next command
                current_command = command
                current_command_result = ""

        elif output.startswith("data:"):
            result_line = output.split("data:")[1].strip()
            current_command_result += result_line + "\n"

    # Store remaining command result
    if current_command_result:
        command_results[current_command] = current_command_result.strip(
        )
        logger.log("Command:", current_command,
                   colors=["GRAY", "INFO"])
        logger.log("Result:", current_command_result,
                   colors=["GRAY", "SUCCESS"])

    return command_results


if __name__ == "__main__":
    commands = [
        "echo 'Hello, world!'",
        "pwd",
        "ls -l",
    ]

    logger.newline()
    logger.debug("Running example 1...")
    command_results = main_run_commands(commands)
    logger.log("example-1 results:", json.dumps(
        command_results, indent=2), colors=["WHITE", "SUCCESS"])

    logger.newline()
    logger.debug("Running example 2...")
    command_results = main_run_commands(commands, work_dir="~/redis")
    logger.log("example-2 results:", json.dumps(
        command_results, indent=2), colors=["WHITE", "SUCCESS"])
