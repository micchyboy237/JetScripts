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
title: Exploring Pragmatic Patterns in Agentic Systems - Nov 04, 2024
---

### Speakers: llan Bigio

### Biography of the speakers:

Ilan Bigio is an engineer on the Developer Experience team at MLX.

### Abstract:

As autonomous AI systems grow increasingly sophisticated, everyone is working to identify optimal architectural patterns. Different design approaches serve distinct priorities – from maximizing agent autonomy to ensuring scalability, maintaining simplicity, or guaranteeing long-term stability. This talk attempts to present a practical taxonomy of these patterns, examining key tradeoffs such as operational range versus reliability, and acceptable failure thresholds. Drawing from our experience developing Swarm, we'll share insights into creating systems that balance reliability, simplicity, and ergonomic design principles.​​​​​​​​​​​​​​​​
"""
logger.info("### Speakers: llan Bigio")

logger.info("\n\n[DONE]", bright=True)