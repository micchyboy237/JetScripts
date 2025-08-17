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
openapi: get /apps
---

<RequestExample>
"""
logger.info("openapi: get /apps")

curl --request GET \
  --url http://localhost:8080/apps

"""
</RequestExample>

<ResponseExample>
"""

{
  "results": [
    {
      "config": "config1.yaml",
      "id": 1,
      "app_id": "app1"
    },
    {
      "config": "config2.yaml",
      "id": 2,
      "app_id": "app2"
    }
  ]
}

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)