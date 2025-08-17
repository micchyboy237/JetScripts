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
title: '📝 Documentation'
description: 'Contribute to Embedchain docs'
---

<Info>
  **Prerequisite** You should have installed Node.js (version 18.10.0 or
  higher).
</Info>

Step 1. Install Mintlify on your OS:

<CodeGroup>
"""
logger.info("title: '📝 Documentation'")

npm i -g mintlify

"""

"""

yarn global add mintlify

"""
</CodeGroup>

Step 2. Go to the `docs/` directory (where you can find `mint.json`) and run the following command:
"""
logger.info("Step 2. Go to the `docs/` directory (where you can find `mint.json`) and run the following command:")

mintlify dev

"""
The documentation website is now available at `http://localhost:3000`.

### Custom Ports

Mintlify uses port 3000 by default. You can use the `--port` flag to customize the port Mintlify runs on. For example, use this command to run in port 3333:
"""
logger.info("### Custom Ports")

mintlify dev --port 3333

"""
You will see an error like this if you try to run Mintlify in a port that's already taken:
"""
logger.info("You will see an error like this if you try to run Mintlify in a port that's already taken:")

Error: listen EADDRINUSE: address already in use :::3000

"""
## Mintlify Versions

Each CLI is linked to a specific version of Mintlify. Please update the CLI if your local website looks different than production.

<CodeGroup>
"""
logger.info("## Mintlify Versions")

npm i -g mintlify@latest

"""

"""

yarn global upgrade mintlify

"""
</CodeGroup>
"""

logger.info("\n\n[DONE]", bright=True)