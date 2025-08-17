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
openapi: post /{app_id}/add
---

<RequestExample>
"""
logger.info("openapi: post /{app_id}/add")

curl --request POST \
  --url http://localhost:8080/{app_id}/add \
  -d "source=https://www.forbes.com/profile/elon-musk" \
  -d "data_type=web_page"

"""
</RequestExample>

<ResponseExample>
"""

{ "response": "fec7fe91e6b2d732938a2ec2e32bfe3f" }

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)