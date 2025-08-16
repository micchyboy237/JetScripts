from jet.logger import CustomLogger
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Contributing

## How to get a notebook displayed on the website

In the notebook metadata set the `tags` and `description` `front_matter` properties. For example:
"""
logger.info("# Contributing")

{
    "...": "...",
    "metadata": {
        "...": "...",
        "front_matter": {
            "tags": ["code generation", "debugging"],
            "description": "Use conversable language learning model agents to solve tasks and provide automatic feedback through a comprehensive example of writing, executing, and debugging Python code to compare stock price changes."
        }
    }
}

"""
**Note**: Notebook metadata can be edited by opening the notebook in a text editor (Or "Open With..." -> "Text Editor" in VSCode)

The `tags` field is a list of tags that will be used to categorize the notebook. The `description` field is a brief description of the notebook.

## Best practices for authoring notebooks

The following points are best practices for authoring notebooks to ensure consistency and ease of use for the website.

- The Colab button will be automatically generated on the website for all notebooks where it is missing. Going forward, it is recommended to not include the Colab button in the notebook itself.
- Ensure the header is a `h1` header, - `#`
- Don't put anything between the yaml and the header

### Consistency for installation and LLM config

You don't need to explain in depth how to install AutoGen. Unless there are specific instructions for the notebook just use the following markdown snippet:

:::info Requirements
Install `autogen`:
"""
logger.info("## Best practices for authoring notebooks")

pip install autogen

"""
For more information, please refer to the [installation guide](/docs/installation/).
:::

Or if extras are needed:

:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:
"""
logger.info("For more information, please refer to the [installation guide](/docs/installation/).")

pip install autogen[retrievechat] flaml[automl]

"""
For more information, please refer to the [installation guide](/docs/installation/).
:::

When specifying the config list, to ensure consistency it is best to use approximately the following code:
"""
logger.info("For more information, please refer to the [installation guide](/docs/installation/).")


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)

"""
Then after the code cell where this is used, include the following markdown snippet:

:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::

## Testing

Notebooks can be tested by running:
"""
logger.info("## Testing")

python website/process_notebooks.py test

"""
This will automatically scan for all notebooks in the notebook/ and website/ dirs.

To test a specific notebook pass its path:
"""
logger.info("This will automatically scan for all notebooks in the notebook/ and website/ dirs.")

python website/process_notebooks.py test notebook/agentchat_logging.ipynb

"""
Options:
- `--timeout` - timeout for a single notebook
- `--exit-on-first-fail` - stop executing further notebooks after the first one fails

### Skip tests

If a notebook needs to be skipped then add to the notebook metadata:
"""
logger.info("### Skip tests")

{
    "...": "...",
    "metadata": {
        "skip_test": "REASON"
    }
}

"""
## Metadata fields

All possible metadata fields are as follows:
"""
logger.info("## Metadata fields")

{
    "...": "...",
    "metadata": {
        "...": "...",
        "front_matter": {
            "tags": "List[str] - List of tags to categorize the notebook",
            "description": "str - Brief description of the notebook",
        },
        "skip_test": "str - Reason for skipping the test. If present, the notebook will be skipped during testing",
        "skip_render": "str - Reason for skipping rendering the notebook. If present, the notebook will be left out of the website.",
        "extra_files_to_copy": "List[str] - List of files to copy to the website. The paths are relative to the notebook directory",
    }
}

logger.info("\n\n[DONE]", bright=True)