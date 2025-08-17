from embedchain import App
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
title: "🧊 Helicone"
description: "Implement Helicone, the open-source LLM observability platform, with Embedchain. Monitor, debug, and optimize your AI applications effortlessly."
"twitter:title": "Helicone LLM Observability for Embedchain"
---

Get started with [Helicone](https://www.helicone.ai/), the open-source LLM observability platform for developers to monitor, debug, and optimize their applications.

To use Helicone, you need to do the following steps.

## Integration Steps

<Steps>
  <Step title="Create an account + Generate an API Key">
    Log into [Helicone](https://www.helicone.ai) or create an account. Once you have an account, you
    can generate an [API key](https://helicone.ai/developer).

    <Note>
      Make sure to generate a [write only API key](helicone-headers/helicone-auth).
    </Note>

  </Step>
  <Step title="Set base_url in the your code">
You can configure your base_url and MLX API key in your codebase
  <CodeGroup>
"""
logger.info("## Integration Steps")


os.environ["OPENAI_API_BASE"] = "https://oai.helicone.ai/{YOUR_HELICONE_API_KEY}/v1"
# os.environ["OPENAI_API_KEY"] = "{YOUR_OPENAI_API_KEY}"

app = App()

app.add("https://en.wikipedia.org/wiki/Elon_Musk")

logger.debug(app.query("How many companies did Elon found? Which companies?"))

"""
</CodeGroup>
  </Step>
<Step title="Now you can see all passing requests through Embedchain in Helicone">
    <img src="/images/helicone-embedchain.png" alt="Embedchain requests" />
  </Step>
</Steps>

Check out [Helicone](https://www.helicone.ai) to see more use cases!
"""
logger.info("Check out [Helicone](https://www.helicone.ai) to see more use cases!")

logger.info("\n\n[DONE]", bright=True)