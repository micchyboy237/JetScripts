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
<CardGroup cols={3}>
  <Card title="Discord" icon="discord" href="https://mem0.dev/DiD" color="#7289DA">
    Join our community
  </Card>
  <Card title="GitHub" icon="github" href="https://github.com/mem0ai/mem0/discussions/new?category=q-a">
    Ask questions on GitHub
  </Card>
  <Card title="Support" icon="calendar" href="https://cal.com/taranjeetio/meet">
  Talk to founders
  </Card>
</CardGroup>
"""
logger.info("Join our community")

logger.info("\n\n[DONE]", bright=True)