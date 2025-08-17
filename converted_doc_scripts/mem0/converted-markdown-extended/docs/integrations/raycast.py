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
title: "Raycast Extension"
description: "Mem0 Raycast extension for intelligent memory management"
---

Mem0 is a self-improving memory layer for LLM applications, enabling personalized AI experiences that save costs and delight users. This extension lets you store and retrieve text snippets using Mem0's intelligent memory system. Find Mem0 in [Raycast Store](https://www.raycast.com/dev_khant/mem0) for using it.

## Getting Started

**Get your API Key**: You'll need a Mem0 API key to use this extension:

a. Sign up at [app.mem0.ai](https://app.mem0.ai)

b. Navigate to your API Keys page

c. Copy your API key

d. Enter this key in the extension preferences

**Basic Usage**:

- Store memories and text snippets
- Retrieve context-aware information
- Manage persistent user preferences
- Search through stored memories

## ✨ Features

**Remember Everything**: Never lose important information - store notes, preferences, and conversations that your AI can recall later

**Smart Connections**: Automatically links related topics, just like your brain does - helping you discover useful connections

**Cost Saver**: Spend less on AI usage by efficiently retrieving relevant information instead of regenerating responses

## 🔑 How This Helps You

**More Personal Experience**: Your AI remembers your preferences and past conversations, making interactions feel more natural

**Learn Your Style**: Adapts to how you work and what you like, becoming more helpful over time

**No More Repetition**: Stop explaining the same things over and over - your AI remembers your context and preferences

---

<Snippet file="get-help.mdx" />
"""
logger.info("## Getting Started")

logger.info("\n\n[DONE]", bright=True)