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
title: Overview
icon: "eye"
iconType: "solid"
---

Welcome to Mem0 Open Source - a powerful, self-hosted memory management solution for AI agents and assistants. With Mem0 OSS, you get full control over your infrastructure while maintaining complete customization flexibility.

We offer two SDKs for Python and Node.js.

Check out our [GitHub repository](https://mem0.dev/gd) to explore the source code.

<CardGroup cols={2}>
<Card title="Python SDK Guide" icon="python" href="/open-source/python-quickstart">
  Learn more about Mem0 OSS Python SDK
</Card>
<Card title="Node.js SDK Guide" icon="node" href="/open-source/node-quickstart">
  Learn more about Mem0 OSS Node.js SDK
</Card>
</CardGroup>

## Key Features

- **Full Infrastructure Control**: Host Mem0 on your own servers
- **Customizable Implementation**: Modify and extend functionality as needed
- **Local Development**: Perfect for development and testing
- **No Vendor Lock-in**: Own your data and infrastructure
- **Community Driven**: Benefit from and contribute to community improvements
"""
logger.info("## Key Features")

logger.info("\n\n[DONE]", bright=True)