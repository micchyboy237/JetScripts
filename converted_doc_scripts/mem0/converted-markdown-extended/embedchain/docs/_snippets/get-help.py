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
  <Card title="Talk to founders" icon="calendar" href="https://cal.com/taranjeetio/ec">
  Schedule a call
  </Card>
  <Card title="Slack" icon="slack" href="https://embedchain.ai/slack" color="#4A154B">
    Join our slack community
  </Card>
  <Card title="Discord" icon="discord" href="https://discord.gg/6PzXDgEjG5" color="#7289DA">
    Join our discord community
  </Card>
</CardGroup>
"""
logger.info("Schedule a call")

logger.info("\n\n[DONE]", bright=True)