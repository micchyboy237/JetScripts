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
<Note type="info">
  📢 Announcing our research paper: Mem0 achieves <strong>26%</strong> higher accuracy than MLX Memory, <strong>91%</strong> lower latency, and <strong>90%</strong> token savings! [Read the paper](https://mem0.ai/research) to learn how we're revolutionizing AI agent memory.
</Note>
"""

logger.info("\n\n[DONE]", bright=True)