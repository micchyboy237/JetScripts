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
  📢 Heads up!
  We're moving to async memory add for a faster experience.
  If you signed up after July 1st, 2025, your add requests will work in the background and return right away.
</Note>
"""
logger.info("We're moving to async memory add for a faster experience.")

logger.info("\n\n[DONE]", bright=True)