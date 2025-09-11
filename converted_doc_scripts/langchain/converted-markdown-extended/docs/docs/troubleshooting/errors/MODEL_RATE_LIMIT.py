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
# MODEL_RATE_LIMIT

You have hit the maximum number of requests that a model provider allows over a given time period and are being temporarily blocked.
Generally, this error is temporary and your limit will reset after a certain amount of time.

## Troubleshooting

The following may help resolve this error:

- Contact your model provider and ask for a rate limit increase.
- If many of your incoming requests are the same, utilize [model response caching](/docs/how_to/chat_model_caching/).
- Spread requests across different providers if your application allows it.
- Use a [`rate_limiter`](/docs/how_to/chat_model_rate_limiting/) to control the rate of requests to the model.
"""
logger.info("# MODEL_RATE_LIMIT")

logger.info("\n\n[DONE]", bright=True)