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
# MODEL_NOT_FOUND

The model name you have specified is not acknowledged by your provider.

## Troubleshooting

The following may help resolve this error:

- Double check the model string you are passing in.
- If you are using a proxy or other alternative host with a model wrapper, confirm that the permitted model names are not restricted or altered.
"""
logger.info("# MODEL_NOT_FOUND")

logger.info("\n\n[DONE]", bright=True)