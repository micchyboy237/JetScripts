from embedchain import App
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
---
title: '📃 JSON'
---

To add any json file, use the data_type as `json`. Headers are included for each line, so for example if you have a json like `{"age": 18}`, then it will be added as `age: 18`.

Here are the supported sources for loading `json`:

1. URL - valid url to json file that ends with ".json" extension.
2. Local file - valid url to local json file that ends with ".json" extension.
3. String - valid json string (e.g. - app.add('{"foo": "bar"}'))

<Tip>
If you would like to add other data structures (e.g. list, dict etc.), convert it to a valid json first using `json.dumps()` function.
</Tip>

## Example

<CodeGroup>
"""
logger.info("## Example")


app = App()

app.add("temp.json")

app.query("What is the net worth of Elon Musk as of October 2023?")

"""

"""

{
    "question": "What is your net worth, Elon Musk?",
    "answer": "As of October 2023, Elon Musk's net worth is $255.2 billion, making him one of the wealthiest individuals in the world."
}

"""
</CodeGroup>
"""

logger.info("\n\n[DONE]", bright=True)