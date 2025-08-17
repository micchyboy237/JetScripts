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
openapi: post /{app_id}/deploy
---


<RequestExample>
"""
logger.info("openapi: post /{app_id}/deploy")

curl --request POST \
  --url http://localhost:8080/{app_id}/deploy \
  -d "api_key=ec-xxxx"

"""
</RequestExample>

<ResponseExample>
"""

{ "response": "App deployed successfully." }

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)