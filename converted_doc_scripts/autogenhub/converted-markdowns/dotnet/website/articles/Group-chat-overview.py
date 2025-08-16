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
@AutoGen.Core.IGroupChat is a fundamental feature in AutoGen. It provides a way to organize multiple agents under the same context and work together to resolve a given task.

In AutoGen, there are two types of group chat:
- @AutoGen.Core.RoundRobinGroupChat : This group chat runs agents in a round-robin sequence. The chat history plus the most recent reply from the previous agent will be passed to the next agent.
- @AutoGen.Core.GroupChat : This group chat provides a more dynamic yet controlable way to determine the next speaker agent. You can either use a llm agent as group admin, or use a @AutoGen.Core.Graph, which is introduced by [this PR](https://github.com/microsoft/autogen/pull/1761), or both to determine the next speaker agent.

> [!NOTE]
> In @AutoGen.Core.GroupChat, when only the group admin is used to determine the next speaker agent, it's recommented to use a more powerful llm model, such as `gpt-4` to ensure the best experience.
"""
logger.info("In AutoGen, there are two types of group chat:")

logger.info("\n\n[DONE]", bright=True)