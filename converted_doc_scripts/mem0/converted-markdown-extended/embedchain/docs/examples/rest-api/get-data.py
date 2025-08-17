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
openapi: get /{app_id}/data
---

<RequestExample>
"""
logger.info("openapi: get /{app_id}/data")

curl --request GET \
  --url http://localhost:8080/{app_id}/data

"""
</RequestExample>

<ResponseExample>
"""

{
  "results": [
    {
      "data_type": "web_page",
      "data_value": "https://www.forbes.com/profile/elon-musk/",
      "metadata": "null"
    }
  ]
}

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)