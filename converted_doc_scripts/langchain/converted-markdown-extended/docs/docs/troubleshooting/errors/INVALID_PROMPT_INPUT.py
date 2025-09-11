from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# INVALID_PROMPT_INPUT

A [prompt template](/docs/concepts/prompt_templates) received missing or invalid input variables.

## Troubleshooting

The following may help resolve this error:

- Double-check your prompt template to ensure that it is correct.
  - If you are using the default f-string format and you are using curly braces `{` anywhere in your template, they should be double escaped like this: `{{` (and if you want to render a double curly brace, you should use four curly braces: `{{{{`).
- If you are using a [`MessagesPlaceholder`](/docs/concepts/prompt_templates/#messagesplaceholder), make sure that you are passing in an array of messages or message-like objects.
  - If you are using shorthand tuples to declare your prompt template, make sure that the variable name is wrapped in curly braces (`["placeholder", "{messages}"]`).
- Try viewing the inputs into your prompt template using [LangSmith](https://docs.smith.langchain.com/) or log statements to confirm they appear as expected.
- If you are pulling a prompt from the [LangChain Prompt Hub](https://smith.langchain.com/prompts), try pulling and logging it or running it in isolation with a sample input to confirm that it is what you expect.
"""
logger.info("# INVALID_PROMPT_INPUT")

logger.info("\n\n[DONE]", bright=True)