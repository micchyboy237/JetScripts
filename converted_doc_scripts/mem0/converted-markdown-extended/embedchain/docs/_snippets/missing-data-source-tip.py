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
<p>If you can't find the specific data source, please feel free to request through one of the following channels and help us prioritize.</p>

<CardGroup cols={2}>
  <Card title="Google Form" icon="file" href="https://forms.gle/NDRCKsRpUHsz2Wcm8" color="#7387d0">
    Fill out this form
  </Card>
  <Card title="Slack" icon="slack" href="https://embedchain.ai/slack" color="#4A154B">
    Let us know on our slack community
  </Card>
  <Card title="Discord" icon="discord" href="https://discord.gg/6PzXDgEjG5" color="#7289DA">
    Let us know on discord community
  </Card>
  <Card title="GitHub" icon="github" href="https://github.com/embedchain/embedchain/issues/new?assignees=&labels=&projects=&template=feature_request.yml" color="#181717">
  Open an issue on our GitHub
  </Card>
  <Card title="Schedule a call" icon="calendar" href="https://cal.com/taranjeetio/ec">
  Schedule a call with Embedchain founder
  </Card>
</CardGroup>
"""
logger.info("Fill out this form")

logger.info("\n\n[DONE]", bright=True)