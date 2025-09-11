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
# Permit

[Permit.io](https://permit.io/) offers fine-grained access control and policy
enforcement. With LangChain, you can integrate Permit checks to ensure only authorized
users can access or retrieve data in your LLM applications.

## Installation and Setup
"""
logger.info("# Permit")

pip install langchain-permit
pip install permit

"""
Set environment variables for your Permit PDP and credentials:
"""
logger.info("Set environment variables for your Permit PDP and credentials:")

export PERMIT_API_KEY="your_permit_api_key"
export PERMIT_PDP_URL="http://localhost:7766"   # or your real PDP endpoint

"""
Make sure your PDP is running and configured. See
[Permit Docs](https://docs.permit.io/sdk/python/quickstart-python/#2-setup-your-pdp-policy-decision-point-container)
for policy setup.

## Tools

See detail on available tools [here](/docs/integrations/tools/permit).

## Retrievers

See detail on available retrievers [here](/docs/integrations/retrievers/permit).
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)