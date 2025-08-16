

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Promptflow

Promptflow is a comprehensive suite of tools that simplifies the development, testing, evaluation, and deployment of LLM based AI applications. It also supports integration with Azure AI for cloud-based operations and is designed to streamline end-to-end development.

Refer to [Promptflow docs](https://autogenhub.github.io/promptflow/) for more information.

Quick links:

- Why use Promptflow - [Link](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow)
- Quick start guide - [Link](https://autogenhub.github.io/promptflow/how-to-guides/quick-start.html)
- Sample application for Promptflow + AutoGen integration - [Link](https://github.com/autogenhub/build-with-autogen/tree/main/samples/apps/promptflow-autogen)

## Sample Flow

![Sample Promptflow](./img/ecosystem-promptflow.png)
"""
logger.info("# Promptflow")

logger.info("\n\n[DONE]", bright=True)