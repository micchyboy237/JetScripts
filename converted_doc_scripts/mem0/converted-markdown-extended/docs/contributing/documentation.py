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
title: Documentation
icon: "book"
---

# Documentation Contributions

## 📌 Prerequisites

Before getting started, ensure you have **Node.js (version 23.6.0 or higher)** installed on your system.

---

## 🚀 Setting Up Mintlify

### Step 1: Install Mintlify

Install Mintlify globally using your preferred package manager:

<CodeGroup>
"""
logger.info("# Documentation Contributions")

npm i -g mintlify

"""

"""

yarn global add mintlify

"""
</CodeGroup>

### Step 2: Run the Documentation Server

Navigate to the `docs/` directory (where `docs.json` is located) and start the development server:
"""
logger.info("### Step 2: Run the Documentation Server")

mintlify dev

"""
The documentation website will be available at: [http://localhost:3000](http://localhost:3000).

---

## 🔧 Custom Ports

By default, Mintlify runs on **port 3000**. To use a different port, add the `--port` flag:
"""
logger.info("## 🔧 Custom Ports")

mintlify dev --port 3333

"""
---

By following these steps, you can efficiently contribute to **Mem0's documentation**. Happy documenting! ✍️
"""
logger.info("By following these steps, you can efficiently contribute to **Mem0's documentation**. Happy documenting! ✍️")

logger.info("\n\n[DONE]", bright=True)