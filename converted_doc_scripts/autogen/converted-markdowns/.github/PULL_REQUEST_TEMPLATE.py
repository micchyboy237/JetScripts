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
<!-- Thank you for your contribution! Please review https://microsoft.github.io/autogen/docs/Contribute before opening a pull request. -->

<!-- Please add a reviewer to the assignee section when you create a PR. If you don't have the access to it, we will shortly find a reviewer and assign them to your PR. -->

## Why are these changes needed?

<!-- Please give a short summary of the change and the problem this solves. -->

## Related issue number

<!-- For example: "Closes #1234" -->

## Checks

- [ ] I've included any doc changes needed for <https://microsoft.github.io/autogen/>. See <https://github.com/microsoft/autogen/blob/main/CONTRIBUTING.md> to build and test documentation locally.
- [ ] I've added tests (if relevant) corresponding to the changes introduced in this PR.
- [ ] I've made sure all auto checks have passed.
"""
logger.info("## Why are these changes needed?")

logger.info("\n\n[DONE]", bright=True)