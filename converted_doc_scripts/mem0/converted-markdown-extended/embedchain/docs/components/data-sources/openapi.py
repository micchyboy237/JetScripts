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
title: 🙌 OpenAPI
---

To add any OpenAPI spec yaml file (currently the json file will be detected as JSON data type), use the data_type as 'openapi'. 'openapi' allows remote urls and conventional file paths.
"""
logger.info("title: 🙌 OpenAPI")


app = App()

app.add("https://github.com/openai/openai-openapi/blob/master/openapi.yaml", data_type="openapi")

app.query("What can MLX API endpoint do? Can you list the things it can learn from?")

"""
<Note>
The yaml file added to the App must have the required OpenAPI fields otherwise the adding OpenAPI spec will fail. Please refer to [OpenAPI Spec Doc](https://spec.openapis.org/oas/v3.1.0)
</Note>
"""
logger.info("The yaml file added to the App must have the required OpenAPI fields otherwise the adding OpenAPI spec will fail. Please refer to [OpenAPI Spec Doc](https://spec.openapis.org/oas/v3.1.0)")

logger.info("\n\n[DONE]", bright=True)