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
# MODEL_AUTHENTICATION

Your model provider is denying you access to their service.

## Troubleshooting

The following may help resolve this error:

- Confirm that your API key or other credentials are correct.
- If you are relying on an environment variable to authenticate, confirm that the variable name is correct and that it has a value set.
  - Note that environment variables can also be set by packages like `dotenv`.
  - For models, you can try explicitly passing an `api_key` parameter to rule out any environment variable issues like this:
"""
logger.info("# MODEL_AUTHENTICATION")

model = ChatOllama(model="llama3.2")

"""
- If you are using a proxy or other custom endpoint, make sure that your custom provider does not expect an alternative authentication scheme.
"""

logger.info("\n\n[DONE]", bright=True)