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
openapi: post /{app_id}/query
---

<RequestExample>
"""
logger.info("openapi: post /{app_id}/query")

curl --request POST \
  --url http://localhost:8080/{app_id}/query \
  -d "query=who is Elon Musk?"

"""
</RequestExample>

<ResponseExample>
"""

{ "response": "Net worth of Elon Musk is $218 Billion." }

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)