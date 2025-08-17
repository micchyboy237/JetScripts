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
openapi: delete /{app_id}/delete
---


<RequestExample>
"""
logger.info("openapi: delete /{app_id}/delete")

curl --request DELETE \
    --url http://localhost:8080/{app_id}/delete

"""
</RequestExample>

<ResponseExample>
"""

{ "response": "App with id {app_id} deleted successfully." }

"""
</ResponseExample>
"""

logger.info("\n\n[DONE]", bright=True)