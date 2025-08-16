from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent and Topic ID Specs

This document describes the structure, constraints, and behavior of Agent IDs and Topic IDs.

## Agent ID

### Required Attributes

#### type

- Type: `string`
- Description: The agent type is not an agent class. It associates an agent with a specific factory function, which produces instances of agents of the same agent `type`. For example, different factory functions can produce the same agent class but with different constructor perameters.
- Constraints: UTF8 and only contain alphanumeric letters (a-z) and (0-9), or underscores (\_). A valid identifier cannot start with a number, or contain any spaces.
- Examples:
  - `code_reviewer`
  - `WebSurfer`
  - `UserProxy`

#### key

- Type: `string`
- Description: The agent key is an instance identifier for the given agent `type`
- Constraints: UTF8 and only contain characters between (inclusive) ascii 32 (space) and 126 (~).
- Examples:
  - `default`
  - A memory address
  - a UUID string

## Topic ID

### Required Attributes

#### type

- Type: `string`
- Description: Topic type is usually defined by application code to mark the type of messages the topic is for.
- Constraints: UTF8 and only contain alphanumeric letters (a-z) and (0-9), ':', '=', or underscores (\_). A valid identifier cannot start with a number, or contain any spaces.
- Examples:
  - `GitHub_Issues`

#### source

- Type: `string`
- Description: Topic source is the unique identifier for a topic within a topic type. It is typically defined by application data.
- Constraints: UTF8 and only contain characters between (inclusive) ascii 32 (space) and 126 (~).
- Examples:
  - `github.com/{repo_name}/issues/{issue_number}`
"""
logger.info("# Agent and Topic ID Specs")

logger.info("\n\n[DONE]", bright=True)