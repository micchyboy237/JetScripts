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
# Contributing to AutoGen

The project welcomes contributions from developers and organizations worldwide. Our goal is to foster a collaborative and inclusive community where diverse perspectives and expertise can drive innovation and enhance the project's capabilities. Whether you are an individual contributor or represent an organization, we invite you to join us in shaping the future of this project. Together, we can build something truly remarkable. Possible contributions include but not limited to:

- Pushing patches.
- Code review of pull requests.
- Documentation, examples and test cases.
- Readability improvement, e.g., improvement on docstr and comments.
- Community participation in [issues](https://github.com/autogenhub/autogen/issues), [discord](https://discord.gg/pAbnFJrkgZ), and [twitter](https://twitter.com/Chi_Wang_).
- Tutorials, blog posts, talks that promote the project.
- Sharing application scenarios and/or related research.

If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

## Roadmaps

To see what we are working on and what we plan to work on, please check our
[Roadmap Issues](https://autogenhub.com/roadmap).

## Becoming a Reviewer

There is currently no formal reviewer solicitation process. Current reviewers identify reviewers from active contributors. If you are willing to become a reviewer, you are welcome to let us know on discord.

## Contact Maintainers

The project is currently maintained by a [dynamic group of volunteers](https://github.com/autogenhub/autogen/blob/main/MAINTAINERS.md) from several organizations. Contact project administrators Chi Wang and Qingyun Wu via auto-gen@outlook.com if you are interested in becoming a maintainer.


## License Headers

To maintain proper licensing and copyright notices, please include the following header at the top of each new source code file you create, regardless of the programming language:
"""
logger.info("# Contributing to AutoGen")



"""
For files that contain or are derived from the original MIT-licensed code from https://github.com/microsoft/autogen, please use this extended header:
"""
logger.info("For files that contain or are derived from the original MIT-licensed code from https://github.com/microsoft/autogen, please use this extended header:")



"""
Please ensure you update the year range as appropriate. If you're unsure which header to use or have any questions about licensing, please don't hesitate to ask in your pull request or reach out to the maintainers.
"""
logger.info("Please ensure you update the year range as appropriate. If you're unsure which header to use or have any questions about licensing, please don't hesitate to ask in your pull request or reach out to the maintainers.")

logger.info("\n\n[DONE]", bright=True)